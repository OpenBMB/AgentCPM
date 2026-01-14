

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import string
from tqdm import tqdm
import shutil
import time
import warnings
from datetime import datetime
import openai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import HLE_JUDGE_PROMPT, JUDGE_PROMPT_BC, JUDGE_PROMPT_XBENCH, SEAL_SCORER_PROMPT, JUDGE_PROMPT_BC_zh

thread_local = threading.local()

# --- LLM Judge trace logging (writes judge raw replies + parse status) ---
JUDGE_TRACE_WRITER: "JudgeTraceWriter | None" = None

class JudgeTraceWriter:
    """Thread-safe markdown logger for LLM-judge raw outputs."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create header once
        if not self.path.exists():
            with self.lock:
                if not self.path.exists():
                    header = "# LLM Judge Trace Log\n\n" + f"- Created at: {datetime.now().isoformat(timespec='seconds')}\n\n"
                    self.path.write_text(header, encoding="utf-8")

    def append(self, md: str) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(md)


def get_client(api_key: str, base_url: str) -> openai.OpenAI:
    """
    Get a separate OpenAI client instance for each thread.
    """
    if not hasattr(thread_local, 'client'):
        thread_local.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return thread_local.client



def get_judge_model(benchmark_name: str) -> str:
    """Return the corresponding judge model based on the dataset name"""
    group = _judge_group(benchmark_name)
    if group == "hle":
        return "o3-mini-2025-01-31"
    else:
        return "gpt-4.1"

LLM_JUDGE_PROMPT_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer. 
Question: {question}
Labeled Answer: {labeled_answer}
Predicted Answer: {pred_answer}
Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text."""

def _normalize_benchmark_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().replace("_", "-")


def _judge_group(benchmark_name: str) -> str:
    n = _normalize_benchmark_name(benchmark_name)


    if n == "hle" or n in {"hle-text", "hle_text"} or n.startswith("hle-") or n.startswith("hle_"):
        return "hle"
    if n in {"browsecomp", "browse-comp", "bc"}:
        return "browsecomp"
    if n in {"browsecomp-zh", "browsecompzh", "bc-zh"}:
        return "browsecomp_zh"
    if n in {"xbench", "x-bench"}:
        return "xbench"
    if n in {"seal-0", "seal0", "webwalker", "frames"}:
        return "seal_family"

    return "seal_family"


def get_judge_prompt_template(benchmark_name: str) -> str:
    group = _judge_group(benchmark_name)
    if group == "hle":
        return HLE_JUDGE_PROMPT
    if group == "browsecomp":
        return JUDGE_PROMPT_BC
    if group == "browsecomp_zh":
        return JUDGE_PROMPT_BC_zh
    if group == "xbench":
        return JUDGE_PROMPT_XBENCH
    return SEAL_SCORER_PROMPT


def get_judge_max_tokens(benchmark_name: str) -> int:
    group = _judge_group(benchmark_name)
    if group in {"browsecomp", "browsecomp_zh"}:
        return 4 
    if group in {"hle", "xbench"}:
        return 4096
    return 20


def parse_judge_verdict(benchmark_name: str, raw_text: str) -> bool | None:
    if not raw_text:
        return None

    group = _judge_group(benchmark_name)
    txt = raw_text.strip()


    if group == "hle":
        m = re.search(r"correct\s*[:：]\s*['\"’“]?\s*(yes|no)\s*['\"’”]?\b", txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower() == "yes"
        return None

    if group in {"browsecomp", "browsecomp_zh"}:
        m = re.search(r"\b([AB])\b", txt)
        if m:
            return m.group(1) == "A"
        return None

 
    if group == "xbench":
        m = re.search(r"结论\s*[:：]\s*(正确|错误)", txt)
        if m:
            return m.group(1) == "正确"

        m2 = re.search(r"\b(correct|incorrect)\b", txt, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).lower() == "correct"
        return None


    low = txt.lower()
    if low == "correct":
        return True
    if low == "incorrect":
        return False


    has_correct = "correct" in low
    has_incorrect = "incorrect" in low
    if has_correct and not has_incorrect:
        return True
    if has_incorrect and not has_correct:
        return False

    return None



def combination(n: int, k: int) -> int:
    """Calculate combination C(n, k)"""
    if k > n or k < 0 or n < 0:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def calculate_pass_at_k_formula(n: int, c: int, k: int) -> float:
    """Pass@k unbiased estimation formula: 1 - C(n-c, k) / C(n, k)"""
    if c == 0: return 0.0
    if c >= n: return 1.0
    if k > n - c: return 1.0
    
    numerator = combination(n - c, k)
    denominator = combination(n, k)
    
    if denominator == 0: return 0.0
    return 1.0 - (numerator / denominator)

def extract_numbers_from_string(text: str) -> List[float]:
 
    if not isinstance(text, str):
        text = str(text)

    text_no_commas = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text_no_commas)
    return [float(num) for num in numbers]

def get_scale_multiplier(text: str) -> float:
    """Return multiplier based on unit words in text."""
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()
    if 'billion' in text_lower:
        return 1_000_000_000
    if 'million' in text_lower:
        return 1_000_000
    if 'thousand' in text_lower:
        return 1_000
    return 1.0 


def _parse_options_from_question(question_text: str) -> Dict[str, str]:
    options = {}
    options_block_match = re.search(r'Options:\s*\n(.*?)$', question_text, re.DOTALL | re.IGNORECASE)
    if not options_block_match: return options
    options_block = options_block_match.group(1)
    option_matches = re.findall(r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)', options_block, re.DOTALL)
    for match in option_matches:
        options[match[0].strip().upper()] = match[1].strip()
    return options

def normalize_number_str(number_str: str) -> float: 
    for char in ["$", "%", ","]: number_str = number_str.replace(char, "")
    try: return float(number_str)
    except (ValueError, TypeError): return float("inf")

def is_float(element: any) -> bool: 
    try: float(element); return True
    except (ValueError, TypeError): return False

def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    if not isinstance(input_str, str):
        input_str = str(input_str)
    
    no_spaces = re.sub(r'\s+', '', input_str)
    lower_case = no_spaces.lower()
    
    if remove_punct:
        
        no_punct = re.sub(r'[^\w.]', '', lower_case)
        return no_punct
        
    return lower_case


def question_scorer(model_answer: str, ground_truth: Any, question_context: str = "", scale: str = "") -> bool:

    if not model_answer or not model_answer.strip():
        return False
    
    model_answer_str, ground_truth_str = str(model_answer), str(ground_truth)


    if re.fullmatch(r'[A-D]', model_answer_str.strip(), re.IGNORECASE):
        options = _parse_options_from_question(question_context)
        option_letter = model_answer_str.strip().upper()
        if option_letter in options:
            model_answer_str = options[option_letter] 
        else:
            return False 

  
    gt_nums = extract_numbers_from_string(ground_truth_str)
    model_nums = extract_numbers_from_string(model_answer_str)
    
    if len(gt_nums) == 1 and len(model_nums) > 0:
        gt_val = gt_nums[0]
        model_val = model_nums[0] 

        
        scale_multiplier = 1.0
        if scale and isinstance(scale, str):
            scale = scale.lower()
            if scale == 'thousand': scale_multiplier = 1000
            elif scale == 'million': scale_multiplier = 1_000_000
            elif scale == 'billion': scale_multiplier = 1_000_000_000
        
        gt_val_scaled = gt_val * scale_multiplier
        
        
        if abs(gt_val_scaled - model_val) < 1e-6:
            return True

    
    if len(gt_nums) == 1 and len(model_nums) == 1:
        gt_val = gt_nums[0]
        model_val = model_nums[0]
        gt_multiplier = get_scale_multiplier(ground_truth_str)
        model_multiplier = get_scale_multiplier(model_answer_str)

        gt_val_scaled = gt_val * gt_multiplier
        model_val_scaled = model_val * model_multiplier
        

        if abs(gt_val_scaled - model_val_scaled) < 1e-6:
            return True
        if abs(gt_val - model_val) < 1e-6 and model_multiplier == 1.0:
            return True

    if any(char in ground_truth_str for char in [",", ";"]):
        gt_elements = [normalize_str(elem, remove_punct=True) for elem in re.split(r'[,;]\s*', ground_truth_str) if elem.strip()]
        normalized_model_answer_for_list = normalize_str(model_answer_str, remove_punct=True)
        
        all_elements_found = True
        for element in gt_elements:
            if element not in normalized_model_answer_for_list:
                all_elements_found = False
                break
        if all_elements_found:
            return True


    normalized_model_answer = normalize_str(model_answer_str, remove_punct=True)
    normalized_ground_truth = normalize_str(ground_truth_str, remove_punct=True)
    
    if normalized_model_answer == normalized_ground_truth:
        return True
    if normalized_model_answer.endswith(normalized_ground_truth) or \
       normalized_model_answer.startswith(normalized_ground_truth):
        return True
            
    return False


def extract_final_answer_advanced(answer_text: str) -> str:
    """
    Extract only the content within <answer>...</answer> tags (preserve original format, no additional cleaning).
    - If multiple <answer> blocks exist (model may give answer then correct), take the last one
    - If no <answer> tags exist, return original text
    """
    if not answer_text or not isinstance(answer_text, str):
        return ""


    pattern_std = r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>"
    matches = re.findall(pattern_std, answer_text)
    if matches:
        return matches[-1].strip()

    pattern_loose = r"(?is)<?\s*answer\s*>(.*?)<?\s*/?\s*answer\s*>"
    matches = re.findall(pattern_loose, answer_text)
    if matches:
        return matches[-1].strip()

    return answer_text.strip()


from pydantic import BaseModel
from typing import Literal

class HLEJudgeParsed(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: bool = True  

def llm_as_judge(
    question: str,
    labeled_answer: str,
    pred_answer: str,
    benchmark_name: str,
    llm_judge_api_key: str,
    llm_judge_api_base: str,
    task_id: str | int = "N/A",
    run_index: int | None = None,
    max_retries: int = 6,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> bool | None:


    group = _judge_group(benchmark_name)
    prompt_template = get_judge_prompt_template(benchmark_name)

    prompt = prompt_template.format(
        question=question,
        correct_answer=labeled_answer,
        response=pred_answer,
        labeled_answer=labeled_answer,
        pred_answer=pred_answer,
    )

    delay = initial_delay
    client = get_client(llm_judge_api_key, llm_judge_api_base)
        # --- Trace logging (optional) ---

    trace_buf: list[str] = []

    def _trace(md: str) -> None:
        # buffer per task to avoid interleaving across threads
        trace_buf.append(md)

    def _flush_trace() -> None:
        if JUDGE_TRACE_WRITER is not None and trace_buf:
            JUDGE_TRACE_WRITER.append("".join(trace_buf))
            trace_buf.clear()


    def _md_code_block(s: str) -> str:
        # Always use 4-backtick fence to avoid conflicts with content
        fence = "````"
        return f"{fence}text\n{s}\n{fence}\n"

    def _md_details(title: str, body: str) -> str:
        return f"<details><summary>{title}</summary>\n\n{_md_code_block(body)}\n</details>\n"

    # One-time header for this judged run
    _trace(f"\n---\n## task_id={task_id} run_index={run_index} benchmark={benchmark_name} group={group}\n")
    _trace(_md_details("Question", question))
    _trace(_md_details("Labeled Answer", labeled_answer))
    _trace(_md_details("Predicted Answer (extracted)", pred_answer))


    for attempt in range(max_retries):
        try:
            _trace(f"\n### Attempt {attempt+1}/{max_retries}\n")

            # 1) Structured-output 
            if group == "hle":
                try:
                    if (
                        hasattr(client, "beta")
                        and hasattr(client.beta, "chat")
                        and hasattr(client.beta.chat, "completions")
                        and hasattr(client.beta.chat.completions, "parse")
                    ):
                        resp = client.beta.chat.completions.parse(
                            model=get_judge_model(benchmark_name),
                            messages=[{"role": "user", "content": prompt}],
                            response_format=HLEJudgeParsed,
                            max_completion_tokens=2048,
                            temperature=1.0,
                            timeout=60.0,
                        )
                        parsed = resp.choices[0].message.parsed
                        if parsed and parsed.correct in {"yes", "no"}:
                            _trace(f"- structured parse success: correct={parsed.correct}\n")
                            _flush_trace()
                            return parsed.correct == "yes"
                except Exception as e_struct:
                    _trace(f"- structured parse failed: {type(e_struct).__name__}: {e_struct}\n")
                    
                    pass

            # 2) Text fallback
            max_tokens = get_judge_max_tokens(benchmark_name)
            resp = client.chat.completions.create(
                model=get_judge_model(benchmark_name),
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=max_tokens,
                timeout=60.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            verdict = parse_judge_verdict(benchmark_name, content)
            _trace(_md_details("Judge raw response", content))
            _trace(f"- parsed verdict: {verdict}\n")


            if (not content) or (verdict is None):
                _trace("- parse failed: empty content or verdict is None; triggering retry\n")
                if group == "hle":
                    _trace("- expected pattern: 'correct: yes' or 'correct: no'\n")
                elif group in {"browsecomp", "browsecomp_zh"}:
                    _trace("- expected pattern: 'A' or 'B'\n")
                raise ValueError(f"Unparseable judge output: {content[:120]!r}")

            _trace(f"- verdict accepted: {verdict}\n")


            _flush_trace()
            return verdict

        except Exception as e:
            _trace(f"- exception: {type(e).__name__}: {e}\n")
            if attempt == max_retries - 1:
                _trace("- giving up: max retries reached\n")
                print(f"Error: LLM Judge API call failed: {e}")
                break
            _trace(f"- retrying after {delay:.2f}s\n")
            time.sleep(delay)
            delay *= backoff_factor
    _trace("\n**Final**: return None (judge failed/unparseable after retries)\n")
    _flush_trace()
    return None



def generate_llm_judge_report(
    correct_list: List[str], incorrect_list: List[Tuple[str, str, str]],
    failed_list: List[Tuple[str, str, str]], output_file: str,
    standard_data: Dict[str, Dict[str, Any]], 
    judgement_map: Dict[str, Dict[int, bool]] = None,
    benchmark_name: str = ""): 
    """Generate special report for LLM as Judge results (integrating Pass@k and Avg@k statistics)."""
    
    report = []
    report.append(f"# Evaluation Report: LLM Judge Special Analysis\n")
    report.append(f"**Model**: {get_judge_model(benchmark_name)}\n")
    

    if judgement_map:
     
        max_k = 0
        for runs in judgement_map.values():
            if runs:
                max_run_index = max(runs.keys())
                max_k = max(max_k, max_run_index + 1)
        
        report.append(f"**Maximum sampling times (Budget)**: {max_k}\n")
        report.append("=" * 60 + "\n")

     
        target_k_values = [1, 2, 4, 8, 16, 32, 64]
        
        report.append("## 📊 1. Pass@k Algorithm Comparison Statistics\n")
        report.append("The table below compares two calculation methods:")
        report.append("- **Naive (First-k)**: Only counts whether there is at least one success in the first k runs (affected by sampling order).")
        report.append("- **Formula (Unbiased)**: Uses all n run data to estimate through formula (unbiased estimation, smoother curve).\n")
        
        report.append("| k (Budget) | Naive Pass@k (%) | Formula Pass@k (%) | Difference |")
        report.append("|---|---|---|---|")

        for k in target_k_values:
            if max_k >= k:
                naive_pass_count = 0
                formula_prob_sum = 0.0
                valid_tasks_count = 0
                
                for task_id, run_dict in judgement_map.items():
                    
                    if not run_dict: continue
                    valid_tasks_count += 1
                    
     
                    first_k_correct = any(run_dict.get(i) is True for i in range(k))
                    if first_k_correct:
                        naive_pass_count += 1
                    
              
                    n_total = len(run_dict)
                    c_total = sum(1 for res in run_dict.values() if res is True)
                    prob = calculate_pass_at_k_formula(n_total, c_total, k)
                    formula_prob_sum += prob

                if valid_tasks_count > 0:
                    naive_acc = (naive_pass_count / valid_tasks_count) * 100
                    formula_acc = (formula_prob_sum / valid_tasks_count) * 100
                    diff = formula_acc - naive_acc
                    report.append(f"| {k} | {naive_acc:.2f} | {formula_acc:.2f} | {diff:+.2f} |")

        report.append("\n" + "-" * 60 + "\n")

        
        report.append(f"## 📈 2. Avg@{max_k} Run Stability Statistics\n")
        
        all_run_scores = []
        avg_table_rows = []
        
      
        for i in range(max_k):
            run_i_correct = 0
            run_i_total = 0
            
            for task_id, run_dict in judgement_map.items():
                if i in run_dict:
                    run_i_total += 1
                    if run_dict[i] is True:
                        run_i_correct += 1
            
            if run_i_total > 0:
                acc_i = (run_i_correct / run_i_total) * 100
                all_run_scores.append(acc_i)
                avg_table_rows.append(f"| Run {i+1} | {run_i_correct}/{run_i_total} | {acc_i:.2f}% |")
        
        if all_run_scores:
            avg_score = sum(all_run_scores) / len(all_run_scores)
            report.append(f"- **Average Accuracy (Avg Accuracy)**: **{avg_score:.2f}%**\n")
            
            report.append("\n| Run Index | Correct/Total | Accuracy |")
            report.append("|---|---|---|")
            
            if len(avg_table_rows) > 10:
                report.extend(avg_table_rows[:5])
                report.append(f"| ... | ... | ... |")
                report.append(avg_table_rows[-1])
            else:
                report.extend(avg_table_rows)
        
        report.append("\n" + "=" * 60 + "\n\n")

    # --- Original detail section ---
    total = len(correct_list) + len(incorrect_list) + len(failed_list)
    accuracy = (len(correct_list) / total) * 100 if total > 0 else 0
    
    report.append("## 3. Detailed Error Analysis")
    report.append(f"- Total evaluation samples: {total}")
    report.append(f"- LLM judgment correct count: {len(correct_list)}")
    report.append(f"- LLM judgment incorrect count: {len(incorrect_list)}")
    report.append(f"- LLM judgment accuracy (Any Correct): {accuracy:.2f}%\n")

    if incorrect_list:
        report.append("### Details of LLM judgments as [Incorrect]")
        for i, (task_id, standard, test) in enumerate(incorrect_list):
            question = standard_data.get(task_id.split('_run_')[0], {}).get("Question", "N/A")
            report.append(f"#### Error {i+1}: {task_id}")
            report.append(f"- Question: {question}")
            report.append(f"- Standard answer: `{standard}`")
            report.append(f"- Model answer: `{test}`\n")

    if failed_list:
        report.append("### Details of LLM [Judgment Failed]")
        for i, (task_id, standard, test) in enumerate(failed_list):
            report.append(f"#### Failed {i+1}: {task_id}")
            report.append(f"- Standard answer: `{standard}`")
            report.append(f"- Model answer: `{test}`\n")

    report_text = "\n".join(report)

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"LLM Judge report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving LLM Judge report: {e}")




def load_test_results(result_dir: str) -> Dict[str, List[Dict[str, Any]]]: 

    results_grouped = {} 
    result_path = Path(result_dir)
    if not result_path.is_dir(): 
        print(f"Error: Result directory does not exist -> {result_dir}");
        return results_grouped
    
    
    task_dirs = [d for d in result_path.iterdir() if d.is_dir()]
    if not task_dirs:
        print(f"Warning: No result subdirectories found in {result_dir}.")
        return results_grouped

    
    new_structure_found = False
    for task_dir in task_dirs:
        task_id = task_dir.name
        
        traj_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("traj_")]
        if traj_dirs:
            new_structure_found = True
            base_task_id = task_id
            
            
            for traj_dir in sorted(traj_dirs, key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 999):
                result_file = traj_dir / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, "r", encoding="utf-8") as f: 
                            result_data = json.load(f)
                        
                        
                        if base_task_id not in results_grouped:
                            results_grouped[base_task_id] = []
                        
                        results_grouped[base_task_id].append(result_data) 
                        
                    except json.JSONDecodeError: 
                        print(f"Warning: Unable to parse JSON file -> {result_file}")

    
    if new_structure_found:
        return results_grouped
        
    
    for task_dir in tqdm(task_dirs, desc="Loading result directories"):
        task_id_with_run = task_dir.name
        
        
        match = re.match(r'^(.*?)_run_(\d+)$', task_id_with_run)
        
        if match:
            base_task_id = match.group(1) # e.g., 'task_1'
        else:
            base_task_id = task_id_with_run # e.g., 'task_1' 

        result_file = task_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f: 
                    result_data = json.load(f)
                
                
                if base_task_id not in results_grouped:
                    results_grouped[base_task_id] = []
                
                results_grouped[base_task_id].append(result_data) 
                
            except json.JSONDecodeError: 
                print(f"Warning: Unable to parse JSON file -> {result_file}")
    
    return results_grouped

def generate_simple_report(report_path: str, benchmark_name: str, total_count: int, correct_count: int):
    """Generate a concise Markdown report containing only core metrics."""
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    report_content = f"""# Evaluation Report: {benchmark_name}

- **Dataset Name**: {benchmark_name}
- **Total Evaluation Samples**: {total_count}
- **Correct Samples**: {correct_count}
- **Final Accuracy**: {accuracy:.2f}%
"""
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Evaluation report generated: {report_path}")
    except Exception as e:
        print(f"[Error] Failed to generate report: {e}")

def process_and_copy_task(task, test_results, llm_judge, judgement_map, success_dir, fail_dir, results_dir_path):
    """
    [Modified] Process all runs of a single base task.
    Now strictly distinguishes each run:
    - Correct run -> success_dir
    - Incorrect run -> fail_dir

    Returns:
    (
      "correct" if at_least_one_pass else "fail",
      task_data (if at_least_one_pass),
      list_of_correct_dialogs
    )
    """
    task_id = task["task_id"]
    
    if task_id not in test_results:
        return ("fail", None, [])  

    all_runs_for_task = test_results[task_id]
    k_count = len(all_runs_for_task)

    at_least_one_pass = False
    correct_dialogs_for_this_task = []

    
    for i, run_result in enumerate(all_runs_for_task):
        run_task_id = f"{task_id}_run_{i}" if k_count > 1 else task_id
        
        
        is_run_correct = False
        
        if llm_judge:
            
            judgement = judgement_map.get(task_id, {}).get(i)
            if judgement is True:
                is_run_correct = True
        else:
            
            conversation_history = run_result.get("conversation_history", [])
            model_raw_answer = ""
            if conversation_history:
                for message in reversed(conversation_history):
                    if message.get("role") == "assistant":
                        model_raw_answer = message.get("content", ""); break
            model_final_answer = extract_final_answer_advanced(model_raw_answer)
            
            if question_scorer(model_final_answer, task["standard_answer"], task["question_text"], task["scale_info"]):
                is_run_correct = True
            elif task["aug_answers"]:
                for aug_ans in task["aug_answers"]:
                    if question_scorer(model_final_answer, aug_ans, task["question_text"], task["scale_info"]):
                        is_run_correct = True
                        break
        
        
        destination_base_dir = success_dir if is_run_correct else fail_dir
        
        
        source_task_dir_new = results_dir_path / task_id / f"traj_{i}"
        
        source_task_dir_old = results_dir_path / run_task_id
        
        source_task_dir = None
        if source_task_dir_new.exists():
            source_task_dir = source_task_dir_new
        elif source_task_dir_old.exists():
            source_task_dir = source_task_dir_old
        else:
            continue  
        
        
        if is_run_correct:
            at_least_one_pass = True
            
            
            source_dialog_path = source_task_dir / "dialog.json"
            if source_dialog_path.exists():
                try:
                    with open(source_dialog_path, 'r', encoding='utf-8') as f_d:
                        d_content = json.load(f_d)
                        correct_dialogs_for_this_task.append({"task_id": run_task_id, "dialog": d_content})
                except Exception:
                    pass

        
        if source_task_dir.exists():
            shutil.copytree(source_task_dir, destination_base_dir / run_task_id, dirs_exist_ok=True)
        
        
        context_index = 1
        while True:
            context_folder_name = f"{run_task_id}-context-{context_index}"
            source_context_dir = results_dir_path / context_folder_name
            
            if source_context_dir.exists() and source_context_dir.is_dir():
                destination_context_dir = destination_base_dir / context_folder_name
                shutil.copytree(source_context_dir, destination_context_dir, dirs_exist_ok=True)
                
                
                if is_run_correct:
                    ctx_dialog_path = source_context_dir / "dialog.json"
                    if ctx_dialog_path.exists():
                        try:
                            with open(ctx_dialog_path, 'r', encoding='utf-8') as f_cd:
                                cd_content = json.load(f_cd)
                                correct_dialogs_for_this_task.append({"task_id": context_folder_name, "dialog": cd_content})
                        except Exception:
                            pass
                
                context_index += 1
            else:
                break

    
    if at_least_one_pass:
        return ("correct", task["task_data"], correct_dialogs_for_this_task)
    else:
        return ("fail", None, [])

def evaluate_and_organize_results(
    ground_truth_path: str, 
    results_dir: str, 
    benchmark_name: str,
    model_name: str,
    output_base_dir: str,
    tasks_output_path: str, 
    dialogs_output_path: str,
    report_output_path: str,
    llm_judge: bool = False,
    llm_judge_report_file: str = None,
    llm_judge_api_key: str = None,
    llm_judge_api_base: str = None
):
    print("Step 1: Loading model run results...");
    test_results = load_test_results(results_dir)
    if not test_results: 
        print("Error: Failed to load any results from result directory");
        return
    print(f"Loaded run results for {len(test_results)} tasks.")
    
    print(f"\nStep 2: Preparing new classification output directories...")
    success_dir = Path(output_base_dir) / benchmark_name / model_name / "success"
    fail_dir = Path(output_base_dir) / benchmark_name / model_name / "fail"
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    print(f" - Success cases will be stored in: {success_dir}")
    print(f" - Failure cases will be stored in: {fail_dir}")

 
    print("\nStep 3.1: Preparing all tasks to be processed...")
    tasks_to_process = []
    standard_data_for_report = {}
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as infile:
            all_tasks = [json.loads(line) for line in infile if line.strip()]
        
        for task_data in all_tasks:
            if not isinstance(task_data, dict):
                print(f"[!!!] Warning: Non-dictionary line found in ground_truth file '{ground_truth_path}': {task_data}. Skipping...")
                continue

            task_id = str(task_data.get("task_id"))
            if task_id not in test_results:
                continue

            standard_answer = ""
            for key in task_data:
                if key.lower() == "final answer": standard_answer = task_data[key]; break
            
            aug_answers = task_data.get("aug_answers", [])
            if isinstance(aug_answers, str) and aug_answers:
                aug_answers = [ans.strip() for ans in aug_answers.split(',')]
            
            question_text = task_data.get("Question", "")
            scale_info = task_data.get("scale", "")
            


            tasks_to_process.append({
                "task_id": task_id,
                "task_data": task_data,
                "standard_answer": str(standard_answer or ""),
                "aug_answers": aug_answers,
                "question_text": str(question_text or ""),
                "scale_info": scale_info,
               
            })
            standard_data_for_report[task_id] = task_data
            
    except FileNotFoundError:
        print(f"Error: Ground truth file not found -> {ground_truth_path}");
        return
    except Exception as e: 
        print(f"Unknown error occurred while preparing tasks: {e}")
        return

    
    llm_judge_results = {"correct": [], "incorrect": [], "failed": []}
    judgement_map = {} # store {task_id: {run_index: judgement}} 

    if llm_judge:
        if not llm_judge_api_key:
            print("\n Enabling LLM Judge (llm_judge=True) requires providing API Key.")
            sys.exit(1)
        
        print(f"\n--- [LLM Judge mode activated] ---")
        print(f"--- Evaluation model: {get_judge_model(benchmark_name)}")
        print(f"--- API Base: {llm_judge_api_base or 'default'}")
        print(f"---------------------------------\n")
        print(f"Step 3.2: Concurrently executing LLM Judge tasks...")
                
        global JUDGE_TRACE_WRITER
        trace_dir = Path(llm_judge_report_file).parent if llm_judge_report_file else Path(report_output_path).parent
        trace_path = trace_dir / "llm_judge_trace.md"
        JUDGE_TRACE_WRITER = JudgeTraceWriter(trace_path)
        print(f"LLM Judge Trace will be recorded to: {trace_path}")


        futures_to_task = {}
        total_runs_to_judge = 0
        with ThreadPoolExecutor(max_workers=30) as executor:
            
            print("Submitting all run instances for all tasks...")
            for task in tasks_to_process:
                task_id = task["task_id"]
                if task_id not in test_results:
                    continue
                
                judgement_map[task_id] = {}
                
                all_runs_for_task = test_results[task_id]
                for i, run_result in enumerate(all_runs_for_task):
                    
                    conversation_history = run_result.get("conversation_history", [])
                    model_raw_answer = ""
                    if conversation_history:
                        for message in reversed(conversation_history):
                            if message.get("role") == "assistant": 
                                model_raw_answer = message.get("content", ""); break
                    model_final_answer = extract_final_answer_advanced(model_raw_answer)
                    
                    
                    future = executor.submit(
                        llm_as_judge,
                        task["question_text"],
                        task["standard_answer"],
                        model_final_answer, 
                        benchmark_name,
                        llm_judge_api_key,
                        llm_judge_api_base,
                        task_id,
                        i,
                    )
                    futures_to_task[future] = (task_id, i, task["standard_answer"], model_final_answer)
                    total_runs_to_judge += 1

            print(f"Total {total_runs_to_judge} run instances submitted for LLM Judge.")
            
            
            print("Collecting evaluation results...")
            for future in tqdm(as_completed(futures_to_task), total=total_runs_to_judge, desc="Running LLM Judge"):
                (task_id, run_index, std_ans, model_ans) = futures_to_task[future]
                try:
                    judgement = future.result()
                    judgement_map[task_id][run_index] = judgement 

                    
                    if judgement is True:
                        llm_judge_results["correct"].append(f"{task_id}_run_{run_index}")
                    elif judgement is False:
                        llm_judge_results["incorrect"].append((f"{task_id}_run_{run_index}", std_ans, model_ans))
                    else: # judgement is None
                        llm_judge_results["failed"].append((f"{task_id}_run_{run_index}", std_ans, model_ans))

                except Exception as e:
                    print(f"Task {task_id}_run_{run_index} execution failed: {e}")
                    judgement_map[task_id][run_index] = None 
                    llm_judge_results["failed"].append((task_id, std_ans, f"Execution Error: {e}"))
    
    print(f"\nStep 3.3: [Concurrent] classifying and copying files for {len(tasks_to_process)} tasks...")
    correct_tasks = [] 
    correct_dialogs = [] 
    total_compared = len(tasks_to_process) 
    results_dir_path = Path(results_dir) 

    
    with ThreadPoolExecutor(max_workers=20) as executor: 
        
        futures_to_task = {}
        
        for task in tasks_to_process:
            future = executor.submit(
                process_and_copy_task, 
                task,
                test_results,
                llm_judge,
                judgement_map,
                success_dir,
                fail_dir,
                results_dir_path 
            )
            futures_to_task[future] = task.get("task_id", "N/A")

        print("All I/O (copy) tasks submitted, collecting results...")
        
        # Collect results
        for future in tqdm(as_completed(futures_to_task), total=len(tasks_to_process), desc="Concurrent copy progress"):
            task_id_for_error = futures_to_task[future]
            try:
                (status, task_data, dialogs_list) = future.result()
                
                
                if status == "correct":
                    if task_data:
                        correct_tasks.append(task_data)
                    if dialogs_list:
                        # Use .extend() to merge lists
                        correct_dialogs.extend(dialogs_list)
                        
            except Exception as e:
                # If a copy task fails, print error but continue
                print(f"\n[Critical Error] Copy thread failed when processing task {task_id_for_error}: {e}")
                import traceback
                traceback.print_exc()

    

       
    

    print(f"\nStep 4: Generating filtered SFT core files...")
    
    with open(tasks_output_path, 'w', encoding='utf-8') as outfile_tasks:
        for task in correct_tasks:
            outfile_tasks.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    with open(dialogs_output_path, 'w', encoding='utf-8') as outfile_dialogs:
        json.dump(correct_dialogs, outfile_dialogs, ensure_ascii=False, indent=2)
    
    print(f"\nStep 5: Generating evaluation report...")
    if llm_judge and llm_judge_report_file:
        print("Generating LLM Judge special report...")
        generate_llm_judge_report(
            llm_judge_results["correct"],
            llm_judge_results["incorrect"],
            llm_judge_results["failed"],
            output_file=llm_judge_report_file,
            standard_data=standard_data_for_report,
            judgement_map=judgement_map,
            benchmark_name=benchmark_name
        )
        print(f"LLM Judge special report saved to: {os.path.abspath(llm_judge_report_file)}")
    elif llm_judge:
        print("Warning: LLM Judge enabled but --llm-judge-report-file path not provided, skipping special report generation.")

    generate_simple_report(
        report_path=report_output_path,
        benchmark_name=benchmark_name,
        total_count=total_compared,
        correct_count=len(correct_tasks)
    )

    print("\n" + "="*50)
    print("Evaluation analysis completed!")
    print(f"Total compared {total_compared} valid tasks.")
    print(f"Among them, {len(correct_tasks)} tasks have correct model answers.")
    print(f"Accuracy: {(len(correct_tasks) / total_compared * 100) if total_compared > 0 else 0:.2f}%")
    print(f"\nClassification output directories generated at: {Path(output_base_dir) / benchmark_name / model_name}")
    print(f"Filtered tasks file saved to: {os.path.abspath(tasks_output_path)}")
    print(f"Filtered dialogs file saved to: {os.path.abspath(dialogs_output_path)}")
    print(f"Evaluation report saved to: {os.path.abspath(report_output_path)}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated evaluation analysis script: filter, classify and generate reports.")
    parser.add_argument('--ground-truth-file', type=str, required=True, help="[Input] Path to .jsonl file containing standard answers.")
    parser.add_argument('--results-dir', type=str, required=True, help="[Input] Root directory containing model run results generated by evaluation script.")
    parser.add_argument('--benchmark-name', type=str, required=True, help="Current benchmark name being evaluated (e.g., gpqa).")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the model being evaluated (e.g., deepseek).")

    parser.add_argument('--output-base-dir', type=str, required=True, help="[Output] Root directory for storing all classification results (e.g., evaluation_outputs).")
    parser.add_argument('--tasks-output-file', type=str, required=True, help="[Output] Path to new .jsonl file for saving filtered tasks.")
    parser.add_argument('--dialogs-output-file', type=str, required=True, help="[Output] Path to new .json file for saving filtered dialogs.")
    parser.add_argument('--report-output-file', type=str, required=True, help="[Output] Path to .md file for saving evaluation report.")
    parser.add_argument('--llm-judge', action='store_true', help="[New] Enable LLM as Judge for scoring, will override 'four-layer funnel' logic.")
    parser.add_argument('--llm-judge-report-file', type=str, help="[New] Path to output .md file for LLM Judge special report.")
    parser.add_argument('--llm-judge-api-key', type=str, help="[New] API Key used by LLM Judge.")
    parser.add_argument('--llm-judge-api-base', type=str, default="https://api.openai.com/v1", help="[New] API Base URL used by LLM Judge.")
    
    args = parser.parse_args()
    
    evaluate_and_organize_results(
        ground_truth_path=args.ground_truth_file, 
        results_dir=args.results_dir, 
        benchmark_name=args.benchmark_name,
        model_name=args.model_name,
        output_base_dir=args.output_base_dir,
        tasks_output_path=args.tasks_output_file, 
        dialogs_output_path=args.dialogs_output_file,
        report_output_path=args.report_output_file,
        llm_judge=args.llm_judge,
        llm_judge_report_file=args.llm_judge_report_file,
        llm_judge_api_key=args.llm_judge_api_key,
        llm_judge_api_base=args.llm_judge_api_base
    )