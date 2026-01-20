#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GAIA Test Result Comparison Script (Integrated Official Scoring Logic + Text Only Task Special Report + LLM as Judge)

Used to compare the results output by the GAIA test model with the standard answers in the original dataset,
calculate the accuracy of the answers, and generate a comparison report.
Integrated official GAIA scoring script logic for more precise fuzzy matching.
Added LLM as Judge functionality for smarter evaluation of text-only tasks.
"""

import os
import sys
import json
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import string
import warnings
import shutil
import openai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



thread_local = threading.local()


def combination(n: int, k: int) -> int:
   
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
    """
    Use formula to calculate Pass@k: ρ(n, c, k) = 1 - C(n-c, k) / C(n, k)
    
    Args:
        n: Total number of samples
        c: Number of correct answers
        k: k in pass@k
    
    Returns:
        Pass@k value (between 0.0 and 1.0)
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n - c:
        # If k > n-c, then C(n-c, k) = 0, so result is 1.0
        return 1.0
    
    numerator = combination(n - c, k)
    denominator = combination(n, k)
    
    if denominator == 0:
        return 0.0
    
    return 1.0 - (numerator / denominator)

def get_client(api_key: str, base_url: str) -> openai.OpenAI:
    """
    Get an independent OpenAI client instance for each thread.
    """
    if not hasattr(thread_local, 'client'):

        thread_local.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return thread_local.client


LLM_JUDGE_MODEL = "gpt-4.1"
LLM_JUDGE_PROMPT_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.
Question: {question}
Labeled Answer: {labeled_answer}
Predicted Answer: {pred_answer}
Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text."""



def normalize_number_str(number_str: str) -> float:
    """Normalize the number string, remove special symbols and convert to float."""
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        
        return float("inf")

def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    """Split the string based on the specified list of delimiters."""
    pattern = f"[{''.join(char_list)}]"
    stripped_s = s.strip().lower()
    return [item.strip() for item in re.split(pattern, stripped_s)]

def is_float(element: any) -> bool:
    
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False

def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalize the string: remove spaces, optionally remove punctuation, convert to lowercase."""
    if not isinstance(input_str, str):
        input_str = str(input_str)
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()

def question_scorer(model_answer: str, ground_truth: str) -> bool:
  
    
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

   
    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)

def check_prediction_contains_answer_letters_in_order(prediction: str, true_answer: str) -> bool:
    
    prediction = prediction.lower()
    true_answer = true_answer.lower()
    if len(prediction) > len(true_answer) * 3:
        return False
    i = 0
    for letter in true_answer:
        if letter in prediction[i:]:
            i += prediction[i:].index(letter) + 1
        else:
            return False
    return True

def check_close_call(prediction: str, true_answer: str, is_correct: bool) -> bool:
    
    if is_correct:
        return True
    else:
        if is_float(true_answer):
            return is_correct
        else:
            if check_prediction_contains_answer_letters_in_order(str(prediction), str(true_answer)) and \
               len(str(true_answer)) * 0.5 <= len(str(prediction)) <= len(str(true_answer)) * 2:
               
                return True
            else:
                return False


def llm_as_judge(question: str, labeled_answer: str, pred_answer: str, api_key: str, api_base: str) -> Optional[bool]:
    """

    Returns:
        - True: If LLM judges as "Correct"
        - False: If LLM judges as "Incorrect"
        - None: If API call fails or return format is inconsistent
    """
    
    
    if pred_answer is None:
        return False
    client = get_client(api_key, api_base)
 

    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        labeled_answer=labeled_answer,
        pred_answer=pred_answer
    )


    
    max_retries = 3
    backoff_factor = 2
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=10,
            )
            content = response.choices[0].message.content.strip().lower()

        

            if "correct" == content:
                return True
            elif "incorrect" == content:
                return False
            else:
                print(f"Warning: LLM returned unparseable content: '{content}'")
                return None 

        except openai.APIError as e:
            print(f"Error: OpenAI API returned error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(delay)
            delay *= backoff_factor
        except Exception as e:
            print(f"Error: Unknown error occurred while calling LLM Judge: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(delay)
            delay *= backoff_factor

    print("Error: LLM Judge API call still failed after multiple retries.")
    return None 



def load_gaia_dataset_jsonl(jsonl_file: str) -> Dict[str, Dict[str, Any]]:
    """Load GAIA dataset (JSONL format) - Robust version"""
    data_items = {}
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Support both 'task_id' and 'id'
                    task_id = str(data.get("task_id") or data.get("id") or "")
                    if task_id:
                        data_items[task_id] = {
                            # Support both 'Final answer' and 'answer', and ensure it's a string
                            "Final answer": str(data.get("Final answer") or data.get("answer") or ""),
                            "Question": str(data.get("Question") or data.get("question") or ""),
                            "level": str(data.get("Level") or data.get("level") or "N/A"),
                            "file_name": str(data.get("file_name") or "")
                        }
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSONL line: {line[:50]}...")
                    continue
    except Exception as e:
        print(f"Error loading GAIA dataset '{jsonl_file}': {e}")
    return data_items

def load_gaia_dataset_json(json_file: str) -> Dict[str, Dict[str, Any]]:
    """Load GAIA dataset (JSON list format) - Robust version"""
    data_items = {}
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for data in data_list:
                # Support both 'id' and 'task_id'
                task_id = str(data.get("id") or data.get("task_id") or "")
                if task_id:
                    data_items[task_id] = {
                        "Final answer": str(data.get("answer") or data.get("Final answer") or ""), 
                        "Question": str(data.get("Question") or data.get("question") or ""),
                        "level": str(data.get("Level") or data.get("level") or "N/A"),
                        "file_name": str(data.get("file_name") or "") 
                    }
    except json.JSONDecodeError as e:
        print(f"JSON parsing error loading GAIA dataset '{json_file}': {e}")
    except Exception as e:
        print(f"Error loading GAIA dataset '{json_file}': {e}")
    return data_items

def load_test_results(result_dir: str) -> Dict[str, List[Dict[str, Any]]]: 
  
 
    results_grouped = {} # Stores { 'task_1': [run_0_data, run_1_data], ... }
    result_path = Path(result_dir)
    if not result_path.is_dir():
        print(f"Error loading test results: directory does not exist {result_path}")
        return results_grouped
        
    try:
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
                            print(f"Warning: Could not parse JSON file -> {result_file}")
                            continue
        
        
        if new_structure_found:
            return results_grouped
        
        
        for task_dir in task_dirs:
            task_id_with_run = task_dir.name
            
            
            match = re.match(r'^(.*?)_run_(\d+)$', task_id_with_run)
            
            if match:
                base_task_id = match.group(1) # e.g., 'task_1'
            else:
                base_task_id = task_id_with_run # e.g., 'task_1' (Pass@1)

            result_file = task_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f: 
                        result_data = json.load(f)
                    
                    if base_task_id not in results_grouped:
                        results_grouped[base_task_id] = []
                    results_grouped[base_task_id].append(result_data)
                        
                except json.JSONDecodeError: 
                    print(f"Warning: Could not parse JSON file -> {result_file}")
                    continue
    except Exception as e:
        print(f"Error loading test results: {e}")
    return results_grouped

def extract_final_answer_advanced(answer_text: str) -> str:
    """Extract final answer from model output using robust split logic."""
    if not answer_text or not isinstance(answer_text, str): 
        return ""

    content = answer_text
    final_answer = None
    
    # Try different tag combinations in priority order
    tag_patterns = [
        ("<answer>", "</answer>"),   # Standard format
        ("<answer>", "</answer"),    # Missing >
        ("<answer>", "/answer>"),    # Missing <
        ("<answer", "</answer>"),    # Start tag missing >
        ("<answer", "</answer"),     # Both missing
        ("answer>", "</answer>"),    # Start tag missing <
        ("answer>", "</answer"),     # Start missing <, end missing >
        ("answer>", "/answer>"),     # Both missing <
        ("answer>", "answer>"),      # Both missing < and /
    ]

    for start_tag, end_tag in tag_patterns:
        # Search for start tag from end to beginning
        search_pos = len(content)
        while search_pos > 0:
            start_idx = content.rfind(start_tag, 0, search_pos)
            if start_idx == -1:
                break
            
            # Check if this start tag is part of an end tag
            is_inside_end_tag = False
            for et in ["</answer>", "</answer", "/answer>"]:
                et_idx = content.rfind(et, 0, start_idx + len(start_tag))
                if et_idx != -1 and et_idx <= start_idx < et_idx + len(et):
                    is_inside_end_tag = True
                    break
            
            if is_inside_end_tag:
                # Continue searching forward
                search_pos = start_idx
                continue
            
            # Found a valid start tag
            content_start = start_idx + len(start_tag)
            remaining_content = content[content_start:]
            
            # Search for end tag
            end_idx = remaining_content.find(end_tag)
            
            if end_idx != -1:
                final_answer = remaining_content[:end_idx].strip()
            else:
                final_answer = remaining_content.strip()
            
            break # Found match with this tag pattern
        
        if final_answer is not None:
            break

    if final_answer is None:
        return None
    
    prefix_match = re.match(r'^\s*[A-D][\.\)\-]\s*', final_answer, re.IGNORECASE)
    if prefix_match:
        final_answer = final_answer[prefix_match.end():].strip()
        
    final_answer = final_answer.replace('}', '').replace('{', '').strip()
    if final_answer.endswith('.'):
        final_answer = final_answer[:-1]
        
    return final_answer.strip()

def compare_answers(standard_data: Dict[str, Dict[str, Any]], 
                    test_results: Dict[str, List[Dict[str, Any]]],
                    pass_k_mode: str = 'any_correct') -> Tuple[List[str], List[str], List[Tuple[str, str, str]], List[str], List[Tuple[str, str, str]]]:
   
    correct = []
    incorrect = []
    error_details = []
    fuzzy_correct = []
    fuzzy_error_details = []

    
    for task_id, data in standard_data.items():
        if task_id not in test_results:
            continue 

        all_runs_for_task = test_results[task_id]
        k_count = len(all_runs_for_task)
        standard_answer_raw = data.get("Final answer", "")
        if not isinstance(standard_answer_raw, str):
            standard_answer_raw = str(standard_answer_raw)

        is_task_correct = False
        is_task_fuzzy_correct = False
        best_failed_answer = "" 

        if pass_k_mode == 'formula':
            
            correct_count = 0  
            fuzzy_correct_count = 0  
            
            for run_result in all_runs_for_task:
                conversation_history = run_result.get("conversation_history", [])
                test_answer_raw = ""
                if conversation_history:
                    for message in reversed(conversation_history):
                        if message.get("role") == "assistant":
                            test_answer_raw = message.get("content", "")
                            break
                
                test_final_answer = extract_final_answer_advanced(test_answer_raw)
                if not test_final_answer:
                    test_final_answer = "[EMPTY]"
                best_failed_answer = test_final_answer
                
                # --- Scoring ---
                is_run_correct = question_scorer(test_final_answer, standard_answer_raw)
                if is_run_correct:
                    correct_count += 1
                else:
                    # Check fuzzy correct
                    is_run_fuzzy_correct = check_close_call(test_final_answer, standard_answer_raw, is_correct=False)
                    if is_run_fuzzy_correct:
                        fuzzy_correct_count += 1
            
            
            pass_at_k_value = calculate_pass_at_k_formula(k_count, correct_count, k_count)
            # If Pass@k value > 0, the task is considered passed
            is_task_correct = pass_at_k_value > 0.0
            
            # If not exactly correct, check if fuzzy correct
            if not is_task_correct and fuzzy_correct_count > 0:
                is_task_fuzzy_correct = True
        
        else:
            # Default mode: any_correct - at least one correct out of k runs
            for run_result in all_runs_for_task:
                conversation_history = run_result.get("conversation_history", [])
                test_answer_raw = ""
                if conversation_history:
                    for message in reversed(conversation_history):
                        if message.get("role") == "assistant":
                            test_answer_raw = message.get("content", "")
                            break
                
                test_final_answer = extract_final_answer_advanced(test_answer_raw)
                if not test_final_answer: # If answer is empty, record it as well
                     test_final_answer = "[EMPTY]"
                best_failed_answer = test_final_answer # Record the last answer
                
                # --- Scoring ---
                is_run_correct = question_scorer(test_final_answer, standard_answer_raw)

                if is_run_correct:
                    is_task_correct = True
                    break # Found one exactly correct, task passed, break inner loop
                
                # If not exactly correct, check fuzzy correct
                is_run_fuzzy_correct = check_close_call(test_final_answer, standard_answer_raw, is_correct=False)
                if is_run_fuzzy_correct:
                    is_task_fuzzy_correct = True
                    # Note: we do *not* break here, because we still want to find an *exactly* correct one in k runs
        
        # --- Inner loop end, determine task status ---
        if is_task_correct:
            correct.append(task_id)
        elif is_task_fuzzy_correct:
            
            fuzzy_correct.append(task_id)
            fuzzy_error_details.append((task_id, standard_answer_raw, best_failed_answer))
        else:
            
            incorrect.append(task_id)
            error_details.append((task_id, standard_answer_raw, best_failed_answer))

    return (
        list(correct), list(incorrect), list(error_details),
        list(fuzzy_correct), list(fuzzy_error_details)
    )

def generate_report(correct_list: List[str], incorrect: List[str], error_details: List[Tuple[str, str, str]], 
                    fuzzy_correct: List[str], fuzzy_error_details: List[Tuple[str, str, str]], 
                    output_file: str = None, standard_data: Dict[str, Dict[str, Any]] = None, report_title: str = "GAIA Test Result Comparison Report"):
    """Generate comparison report"""
    total = len(correct_list) + len(incorrect) + len(fuzzy_correct)
    accuracy = (len(correct_list) / total) * 100 if total > 0 else 0
    fuzzy_accuracy = ((len(correct_list) + len(fuzzy_correct)) / total) * 100 if total > 0 else 0
    
    level_stats = {}
    for task_id in correct_list + incorrect + fuzzy_correct:
        level = standard_data.get(task_id, {}).get("level", "N/A")
        if level not in level_stats:
            level_stats[level] = {"total": 0, "correct": 0, "fuzzy_correct": 0}
        level_stats[level]["total"] += 1
        if task_id in correct_list:
            level_stats[level]["correct"] += 1
        elif task_id in fuzzy_correct:
            level_stats[level]["fuzzy_correct"] += 1
    
    report = []
    report.append(f"# {report_title} (Integrated Official Scoring Logic)")
    report.append(f"\n## Overall Statistics")
    report.append(f"- Total samples: {total}")
    report.append(f"- Exactly correct count: {len(correct_list)}")
    report.append(f"- Fuzzy correct count: {len(fuzzy_correct)}")
    report.append(f"- Incorrect count: {len(incorrect)}")
    report.append(f"- Exactly correct rate: {accuracy:.2f}%")
    report.append(f"- Fuzzy correct rate (including exactly correct): {fuzzy_accuracy:.2f}%\n")
    
    report.append("## Statistics by Difficulty Level\n")
    report.append("| Difficulty Level | Total | Exactly Correct | Exact Rate | Fuzzy Correct | Fuzzy Rate |")
    report.append("|----------|--------|------------|------------|------------|------------|")
    
    for level, stats in sorted(level_stats.items()):
        level_total = stats["total"]
        level_correct = stats["correct"]
        level_fuzzy = stats["fuzzy_correct"]
        level_correct_rate = (level_correct / level_total) * 100 if level_total > 0 else 0
        level_fuzzy_rate = ((level_correct + level_fuzzy) / level_total) * 100 if level_total > 0 else 0
        report.append(f"| {level} | {level_total} | {level_correct} | {level_correct_rate:.2f}% | {level_fuzzy} | {level_fuzzy_rate:.2f}% |")
    
    report.append("")
    
    if error_details:
        report.append("\n## Error Details")
        for i, (task_id, standard, test) in enumerate(error_details):
            question = standard_data.get(task_id, {}).get("Question", "N/A")
            level = standard_data.get(task_id, {}).get("level", "N/A")
            report.append(f"### Error {i+1}: {task_id}")
            report.append(f"Question: {question}")
            report.append(f"Difficulty Level: {level}")
            report.append(f"Standard Answer: {standard}")
            report.append(f"Test Answer: {test}\n")
    
    if fuzzy_error_details:
        report.append("\n## Fuzzy Correct Details")
        for i, (task_id, standard, test) in enumerate(fuzzy_error_details):
            question = standard_data.get(task_id, {}).get("Question", "N/A")
            level = standard_data.get(task_id, {}).get("level", "N/A")
            report.append(f"### Fuzzy Correct {i+1}: {task_id}")
            report.append(f"Question: {question}")
            report.append(f"Difficulty Level: {level}")
            report.append(f"Standard Answer: {standard}")
            report.append(f"Test Answer: {test}\n")
    
    report.append("\n## List of Correct Answers\n")
    report.append(", ".join(correct_list))
    
    report_text = "\n".join(report)
    
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")

def generate_llm_judge_report(
    correct_list: List[str], incorrect_list: List[Tuple[str, str, str]],
    failed_list: List[Tuple[str, str, str]], output_file: str,
    standard_data: Dict[str, Dict[str, Any]], judgement_map: Dict[str, List[Optional[bool]]] = None):
    """Generate dedicated report for LLM as Judge results."""
    total = len(correct_list) + len(incorrect_list) + len(failed_list)
    accuracy = (len(correct_list) / total) * 100 if total > 0 else 0

    report = []
    if judgement_map:
        max_k = max((len(runs) for runs in judgement_map.values()), default=0)
        
        
        report.append(f"# GAIA LLM Judge Multi-sampling Evaluation Report\n")
        report.append(f"**Model**: {LLM_JUDGE_MODEL}\n")
        report.append(f"**Total Sampling Count**: {max_k}\n\n")
        report.append("=" * 60 + "\n")
        
        
        target_k_values = [1, 2, 4, 8, 16, 32, 64]
        
        report.append("## 📊 Pass@k Algorithm Comparison Statistics\n")
        report.append("The following table compares two calculation methods:")
        report.append("- **Naive (First-k)**: Only counts whether at least one success occurred in the first k runs (influenced by sampling order, high fluctuation).")
        report.append("- **Formula (Unbiased)**: Uses all n runs data, estimated by formula $1 - \\binom{n-c}{k}/\\binom{n}{k}$ (unbiased estimate, smoother curve).\n")
        
        report.append("| k (Budget) | Naive Pass@k (%) | Formula Pass@k (%) | Difference (Formula - Naive) |")
        report.append("|---|---|---|---|")

        for k in target_k_values:
            if max_k >= k:
                # 1. Counter: Naive method
                naive_pass_count = 0
                
                # 2. Accumulator: Formula method (accumulated probability)
                formula_prob_sum = 0.0
                
                valid_tasks_count = len(judgement_map)
                
                for task_id, judgements in judgement_map.items():
                    
                    first_k_runs = judgements[:k]
                    if any(j is True for j in first_k_runs):
                        naive_pass_count += 1
                    
                    
                    n_total = len(judgements)
                    c_total = sum(1 for j in judgements if j is True)
                    
                    prob = calculate_pass_at_k_formula(n_total, c_total, k)
                    formula_prob_sum += prob

                if valid_tasks_count > 0:
                   
                    naive_acc = (naive_pass_count / valid_tasks_count) * 100
                    formula_acc = (formula_prob_sum / valid_tasks_count) * 100
                    diff = formula_acc - naive_acc
                    
                    
                    report.append(f"| {k} | {naive_acc:.2f} | {formula_acc:.2f} | {diff:+.2f} |")
        
        report.append("\n" + "=" * 60 + "\n\n")
        
        
        report.append("\n" + "=" * 60 + "\n\n")
    report.append(f"# GAIA Special Test Report (LLM as Judge - Model: {LLM_JUDGE_MODEL})\n")
    report.append("## Overall Statistics")
    report.append(f"- Total evaluated samples: {total}")
    report.append(f"- LLM judge correct count: {len(correct_list)}")
    report.append(f"- LLM judge incorrect count: {len(incorrect_list)}")
    report.append(f"- LLM judge failed count (API or format issues): {len(failed_list)}")
    report.append(f"- LLM judge accuracy: {accuracy:.2f}%\n")

    if incorrect_list:
        report.append("## Details of LLM Judging as [Incorrect]")
        for i, (task_id, standard, test) in enumerate(incorrect_list):
            question = standard_data.get(task_id, {}).get("Question", "N/A")
            level = standard_data.get(task_id, {}).get("level", "N/A")
            report.append(f"### Error {i+1}: {task_id}")
            report.append(f"- Question: {question}")
            report.append(f"- Difficulty Level: {level}")
            report.append(f"- Standard Answer: `{standard}`")
            report.append(f"- Model Answer: `{test}`\n")

    if failed_list:
        report.append("## Details of LLM [Judging Failure]")
        for i, (task_id, standard, test) in enumerate(failed_list):
            question = standard_data.get(task_id, {}).get("Question", "N/A")
            level = standard_data.get(task_id, {}).get("level", "N/A")
            report.append(f"### Failure {i+1}: {task_id}")
            report.append(f"- Question: {question}")
            report.append(f"- Difficulty Level: {level}")
            report.append(f"- Standard Answer: `{standard}`")
            report.append(f"- Model Answer: `{test}`\n")

    report.append("## List of LLM Judging as [Correct]\n")
    report.append(", ".join(correct_list))

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


def organize_results_by_run_granularity(
    result_dir: str,
    output_report_file: str,
    test_results: Dict[str, List[Dict[str, Any]]],
    standard_data: Dict[str, Dict[str, Any]],
    judgement_map: Dict[str, List[Optional[bool]]] = None
):
    """
    Organize files by copying each run to success or fail directory based on its correctness.
    Supports both Rule-Based and LLM Judge decision modes.
    """
    if not output_report_file:
        print("No output file specified, skipping file organization.")
        return

    # Determine output base directory
    base_output_dir = os.path.dirname(output_report_file)
    success_dir = os.path.join(base_output_dir, "success")
    fail_dir = os.path.join(base_output_dir, "fail")

    # Create directories
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    print(f"\nOrganizing files by trajectory (Run) granularity...")
    print(f" - Success Directory: {success_dir}")
    print(f" - Fail Directory:    {fail_dir}")

    copy_count = 0

    result_dir_path = Path(result_dir)

    def _copy_run_and_context(src_path: Path, dest_base: Path, target_folder_name: str) -> bool:
        """
        Copy a single run folder + all its associated -context- folders
        Returns True if the main run folder was copied successfully (used for copy_count statistics)
        """
        dest_path = dest_base / target_folder_name
        try:
            
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)

            
            context_idx = 1
            while True:
                ctx_name = f"{src_path.name}-context-{context_idx}"
                src_ctx = result_dir_path / ctx_name
                if not src_ctx.exists():
                    break

                dest_ctx = dest_base / ctx_name
                if dest_ctx.exists():
                    shutil.rmtree(dest_ctx)
                shutil.copytree(src_ctx, dest_ctx)

                context_idx += 1

            return True

        except Exception as e:
            print(f"Copy {src_path.name} failed: {e}")
            return False

    futures = []

    
    with ThreadPoolExecutor(max_workers=20) as executor:
        for task_id, runs in tqdm(test_results.items(), desc="Organizing Files"):
            # Get standard answer data
            std_info = standard_data.get(task_id, {})
            std_ans = str(std_info.get("Final answer", ""))

            for run_index, run_data in enumerate(runs):
                # 1. Determine if this run is correct (preserving your original logic)
                is_run_correct = False

                if judgement_map:
                    if task_id in judgement_map and run_index < len(judgement_map[task_id]):
                        if judgement_map[task_id][run_index] is True:
                            is_run_correct = True
                else:
                    conversation_history = run_data.get("conversation_history", [])
                    test_answer_raw = ""
                    if conversation_history:
                        for message in reversed(conversation_history):
                            if message.get("role") == "assistant":
                                test_answer_raw = message.get("content", "")
                                break

                    test_final_answer = extract_final_answer_advanced(test_answer_raw)

                    if question_scorer(test_final_answer, std_ans):
                        is_run_correct = True
                    elif check_close_call(test_final_answer, std_ans, is_correct=False):
                        is_run_correct = True

                
                src_path_new = result_dir_path / task_id / f"traj_{run_index}"
                
                src_path_old_multi = result_dir_path / f"{task_id}_run_{run_index}"
                src_path_old_single = result_dir_path / task_id
                
                src_path = None
                target_folder_name = f"{task_id}_run_{run_index}"  
                
                
                if src_path_new.exists():
                    src_path = src_path_new
                
                elif src_path_old_multi.exists():
                    src_path = src_path_old_multi
                    target_folder_name = f"{task_id}_run_{run_index}"
                elif run_index == 0 and src_path_old_single.exists():
                    src_path = src_path_old_single
                    target_folder_name = task_id
                else:
                    continue  

                
                dest_base = Path(success_dir) if is_run_correct else Path(fail_dir)
                futures.append(
                    executor.submit(_copy_run_and_context, src_path, dest_base, target_folder_name)
                )

        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Copying Folders"):
            try:
                if fut.result():
                    copy_count += 1
            except Exception as e:
                print(f"Concurrent copy task exception: {e}")


    print(f"File organization complete, processed {copy_count} trajectory folders.")

def main():
    parser = argparse.ArgumentParser(description="GAIA Test Result Comparison Tool (Integrated Official Scoring Logic and LLM as Judge)")
    parser.add_argument("--data", default="./GAIA/2023/validation/metadata.jsonl", help="Path to original GAIA dataset JSONL file for [Overall Report] evaluation")
    parser.add_argument("--text-data", help="Path to specific GAIA dataset file (.json or .jsonl) for [Text Only Tasks] and [LLM Judge] evaluation")
    parser.add_argument("--result-dir", required=True, help="Path to test results directory")
    parser.add_argument("--output", help="Output path for [Overall Report] file (optional)")
    parser.add_argument("--output-text-only", help="Output path for [Text Only Task Report] file (optional)")
    
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM as Judge evaluation for text only tasks")
    parser.add_argument("--output-llm-judge", help="Output path for [LLM Judge Report] file (optional)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key. Recommended to use OPENAI_API_KEY environment variable")
    parser.add_argument("--api-base", default="https://toollearning.cn/v1/", help="OpenAI API base URL, for compatibility with third-party services")
    parser.add_argument("--pass-k-mode", type=str, default='any_correct', choices=['any_correct', 'formula'],
                        help="[New] Pass@k calculation mode: 'any_correct'=at least one correct in k runs (default), 'formula'=uses formula ρ(n,c,k)=1-C(n-c,k)/C(n,k)")

    args = parser.parse_args()
    
    print("--- Initialization ---")
    print(f"Loading GAIA dataset for overall report: {args.data}")
    gaia_data_full = load_gaia_dataset_jsonl(args.data)
    print(f"Loaded {len(gaia_data_full)} GAIA standard data items")

    print(f"Loading test results: {args.result_dir}")
    test_results = load_test_results(args.result_dir)
    print(f"Loaded {len(test_results)} model test results")
    
    print("\n--- 1. Starting [Overall Report] Generation (Evaluating all tasks) ---")
    full_data_to_compare = {tid: data for tid, data in gaia_data_full.items() if tid in test_results}
    print(f"Will perform comprehensive evaluation on {len(full_data_to_compare)} tasks...")
    comparison_results_full = compare_answers(full_data_to_compare, test_results, pass_k_mode=args.pass_k_mode)
    correct_full, incorrect_full, error_details_full, fuzzy_correct_full, fuzzy_error_details_full = comparison_results_full
    output_file_full = args.output or f"gaia_comparison_report_ALL_TASKS.md"
    generate_report(correct_full, incorrect_full, error_details_full, 
                    fuzzy_correct_full, fuzzy_error_details_full, 
                    output_file_full, gaia_data_full, report_title="GAIA Comprehensive Test Report (All Tasks)")

    if args.text_data:
        print(f"\nLoading specific dataset for text evaluation: {args.text_data}")
        if args.text_data.endswith('.json'):
            gaia_data_text = load_gaia_dataset_json(args.text_data)
        else:
            gaia_data_text = load_gaia_dataset_jsonl(args.text_data)
        print(f"Loaded {len(gaia_data_text)} data items for text evaluation")
    else:
        print("\nNo specific text dataset provided, filtering from main dataset.")
        gaia_data_text = gaia_data_full

    print("\n--- 2. Starting [Text Only Task Report] Generation (Rule-based) ---")
    text_only_gaia_data_full = {
        task_id: data for task_id, data in gaia_data_text.items()
        if data.get("file_name", "") == ""
    }
    text_only_data_to_compare = {tid: data for tid, data in text_only_gaia_data_full.items() if tid in test_results}
    print(f"Found {len(text_only_gaia_data_full)} text-only tasks, {len(text_only_data_to_compare)} found in test results, proceeding with evaluation...")

    if text_only_data_to_compare:
        comparison_results_text = compare_answers(text_only_data_to_compare, test_results, pass_k_mode=args.pass_k_mode)
        correct_text, incorrect_text, error_details_text, fuzzy_correct_text, fuzzy_error_details_text = comparison_results_text
        
        if args.output_text_only:
            output_file_text = args.output_text_only
        else:
            base, ext = os.path.splitext(output_file_full)
            output_file_text = f"{base}_TEXT_ONLY{ext}"
            
        generate_report(correct_text, incorrect_text, error_details_text,
                        fuzzy_correct_text, fuzzy_error_details_text,
                        output_file_text, gaia_data_text, report_title="GAIA Special Test Report (Text Only Tasks - Rule-based)")
    else:
        print("No text-only tasks found for evaluation. Skipping special report generation.")

    if args.llm_judge:
        print("\n--- 3. Starting [LLM as Judge] Evaluation ---")

        if not args.api_key:
            print("Error: API key not found. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
            sys.exit(1)
        
        if not full_data_to_compare:
            print("No tasks available for LLM evaluation, skipping this step.")
            return

        llm_correct = []
        llm_incorrect_details = []
        llm_failed_details = []

        print(f"Will use model '{LLM_JUDGE_MODEL}' to concurrently evaluate all run instances of {len(full_data_to_compare)} base tasks...")
        
        futures = {}
        total_runs_to_judge = 0
        
        with ThreadPoolExecutor(max_workers=30) as executor: 
            
            print("Submitting all LLM Judge tasks...")
            
            
            for task_id, data in full_data_to_compare.items():
                if task_id not in test_results:
                    continue

                question = data.get("Question", "")
                standard_answer = str(data.get("Final answer", ""))
                
                all_runs_for_task = test_results[task_id]
                
            
                for i, run_result in enumerate(all_runs_for_task):
                    test_answer_raw = ""
                    conversation_history = run_result.get("conversation_history", [])
                    if conversation_history:
                        for message in reversed(conversation_history):
                            if message.get("role") == "assistant":
                                test_answer_raw = message.get("content", "")
                                break
                    
                    pred_answer = extract_final_answer_advanced(test_answer_raw)
                    
                    future = executor.submit(
                        llm_as_judge, 
                        question, 
                        standard_answer, 
                        pred_answer, 
                        args.api_key,
                        args.api_base
                    )
                    # Stores (base task ID, run index, standard answer, predicted answer)
                    futures[future] = (task_id, i, standard_answer, pred_answer)
                    total_runs_to_judge += 1
        
            print(f"Total of {total_runs_to_judge} run instances submitted for LLM Judge.")
            
            # --- Collect results ---
            
            judgement_map = {} 
            
            all_run_details = {} 
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running LLM Judge"):
                (task_id, run_index, std_ans, model_ans) = futures[future]
                
                if task_id not in judgement_map:
                    judgement_map[task_id] = []
                    all_run_details[task_id] = []
                    
                try:
                    judgement = future.result() 
                    judgement_map[task_id].append(judgement)
                    all_run_details[task_id].append((std_ans, model_ans))
                    
                    if judgement is None: # API call successful but returned unparseable content
                         llm_failed_details.append((f"{task_id}_run_{run_index}", std_ans, model_ans))

                except Exception as e:
                    print(f"Task {task_id}_run_{run_index} execution failed: {e}")
                    judgement_map[task_id].append(None) # None indicates failure
                    all_run_details[task_id].append((std_ans, model_ans))
                    llm_failed_details.append((f"{task_id}_run_{run_index}", std_ans, f"Execution Error: {e}"))
            
            # --- Aggregate Pass@k results ---
            print("LLM Judge evaluation complete, aggregating Pass@k results...")
            for task_id, judgements in judgement_map.items():
                
                if any(j is True for j in judgements):
                    llm_correct.append(task_id)
                else:
                    
                    runs_details = all_run_details.get(task_id, [])
                    
                    
                    std_ans_report = runs_details[0][0] if runs_details else "N/A"
                    
                    model_ans_report = "N/A (Run details not found)"
                    if runs_details:
                        
                        model_ans_report = runs_details[0][1]

                    
                    first_incorrect_ans = None
                    for i, j in enumerate(judgements):
                        if j is False: # Found a definite "Incorrect"
                            if i < len(runs_details):
                                first_incorrect_ans = runs_details[i][1] # Get model answer for this run
                                break
                    
                    if first_incorrect_ans:
                        model_ans_report = first_incorrect_ans
                    
                    # Add (task_id, standard answer, specific model answer) to error list
                    llm_incorrect_details.append((task_id, std_ans_report, model_ans_report))

            
            
            
                    
        print("LLM as Judge evaluation complete.")

        
        if args.output_llm_judge:
            output_file_llm = args.output_llm_judge
        else:
            base, ext = os.path.splitext(output_file_full)
            output_file_llm = f"{base}_LLM_JUDGE{ext}"

        generate_llm_judge_report(
            llm_correct, llm_incorrect_details, llm_failed_details,
            output_file_llm, gaia_data_text, judgement_map=judgement_map
        )

        print("\n--- 3.1. Starting [LLM Judge] Avg@k Mean Score Generation ---")
        max_k_llm = 0

        if judgement_map:
            max_k_llm = max((len(runs) for runs in judgement_map.values()), default=0)

        if max_k_llm > 0:
            all_run_scores_llm = []
            run_details_for_report = []  # [New] Used to store detailed info for each run
            
            for i in range(max_k_llm): # Traverse k runs
                run_i_correct = 0
                run_i_incorrect = 0
                run_i_failed = 0
                
                # Traverse all tasks
                for task_id, judgements in judgement_map.items():
                    if i < len(judgements):
                        judgement = judgements[i]
                        if judgement is True:
                            run_i_correct += 1
                        elif judgement is False:
                            run_i_incorrect += 1
                        else: 
                            run_i_failed += 1
                
                total_i = run_i_correct + run_i_incorrect + run_i_failed
                if total_i > 0:
                    accuracy_i = (run_i_correct / total_i) * 100
                    all_run_scores_llm.append(accuracy_i)
                    
                    
                    run_details_for_report.append({
                        "run_index": i + 1,
                        "correct": run_i_correct,
                        "incorrect": run_i_incorrect,
                        "failed": run_i_failed,
                        "total": total_i,
                        "accuracy": accuracy_i
                    })
                    
                    print(f"    - Run {i+1}/{max_k_llm} (LLM Judge): Accuracy={accuracy_i:.2f}% (Total {total_i} questions)")

            if all_run_scores_llm:
                avg_acc_llm = sum(all_run_scores_llm) / len(all_run_scores_llm)
                
                if args.output_llm_judge:
                    output_file_llm = args.output_llm_judge
                else:
                    base, ext = os.path.splitext(output_file_full)
                    output_file_llm = f"{base}_LLM_JUDGE{ext}"

                
                avg_summary_lines = [
                    f"\n\n" + "="*50 + "\n",
                    f"## Avg@{max_k_llm} Final Mean Score (LLM as Judge)\n",
                    f"- **Mean Accuracy: {avg_acc_llm:.2f}%**\n",
                    f"\n### Detailed Results per Run\n",
                    f"\n| Run | Correct | Incorrect | Failed | Total | Accuracy |",
                    f"\n|-----|---------|-----------|--------|-------|----------|"
                ]
                
                for detail in run_details_for_report:
                    avg_summary_lines.append(
                        f"\n| Run {detail['run_index']} | {detail['correct']} | "
                        f"{detail['incorrect']} | {detail['failed']} | "
                        f"{detail['total']} | {detail['accuracy']:.2f}% |"
                    )
                
                avg_summary_lines.append(
                    f"\n\n(The above mean score is the average of {len(all_run_scores_llm)} individual runs)\n"
                )
                
                avg_summary = "".join(avg_summary_lines)
                print(avg_summary)
                try:
                   
                    pass_k_content = ""
                    if os.path.exists(output_file_llm):
                        with open(output_file_llm, "r", encoding="utf-8") as f_read:
                            pass_k_content = f_read.read()

                   
                    with open(output_file_llm, "w", encoding="utf-8") as f_write:
                        
                        f_write.write(avg_summary)
                        f_write.write("\n\n" + "="*50 + "\n\n") 
                        
                       
                        f_write.write(pass_k_content)
                    
                    print(f"Avg@k summary added to the top of {output_file_llm}.")
                    
                except Exception as e:
                    print(f"Warning: Failed to write Avg@k summary to top of LLM Judge report: {e}")

    final_output_file = args.output_llm_judge if args.llm_judge else (args.output or args.output_text_only)
  
    if not final_output_file:
         final_output_file = os.path.join(args.result_dir, "report.md")

    
    current_judgement_map = locals().get('judgement_map', None) if args.llm_judge else None
    
    organize_results_by_run_granularity(
        result_dir=args.result_dir,
        output_report_file=final_output_file,
        test_results=test_results,
        standard_data=gaia_data_full, 
        judgement_map=current_judgement_map
    )

if __name__ == "__main__":
    main()
