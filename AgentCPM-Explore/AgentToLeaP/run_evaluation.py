import sys
import argparse
import json
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count
import subprocess
from functools import partial

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir 
sys.path.append(str(evaluation_dir))


from data_test_copy import GaiaApiTest

def run_worker_task(tasks: List[Dict[str, Any]], worker_id: int, common_args: argparse.Namespace):
    """
    This is the task function to be executed by each independent process (worker).
    """
    
    # Only process 0 can print to the terminal; other processes remain silent.
    if worker_id != 0:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')
        import logging
        logger_names = list(logging.Logger.manager.loggerDict.keys()) + [""]
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.handlers = [] 
            logger.addHandler(logging.NullHandler()) 
            logger.propagate = False 

    
    print(f"[Process {worker_id}] Starting to process {len(tasks)} tasks...")

    
    temp_dir = Path(common_args.output_dir) / "temp_inputs"
    os.makedirs(temp_dir, exist_ok=True)
    worker_input_file = temp_dir / f"worker_{worker_id}_input.jsonl"
    
    with open(worker_input_file, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    
    async def run_test():

        

        test_instance = GaiaApiTest(
            # --- Main Model Parameters ---
            main_provider=common_args.provider,
            main_model=common_args.model,
            main_api_key=common_args.api_key,
            main_base_url=common_args.base_url,
            main_temperature=common_args.temperature,
            main_top_p=common_args.top_p,
            main_presence_penalty=common_args.presence_penalty,
            main_max_tokens=common_args.max_tokens,
            # --- Processor Model Parameters ---
            processor_provider=common_args.processor_provider,
            processor_model=common_args.processor_model,
            processor_api_key=common_args.processor_api_key,
            processor_base_url=common_args.processor_base_url,
            # --- Passing Flags ---
            use_browser_processor=common_args.use_browser_processor,
            return_thought=common_args.return_thought,
            use_context_manager=common_args.use_context_manager,
            # --- General Parameters ---
            manager_url=common_args.manager_url,
            max_interactions=common_args.max_interactions,
            output_dir=str(common_args.output_dir),
            files_dir=common_args.files_dir,
            tool_start_tag=common_args.tool_start_tag,
            tool_end_tag=common_args.tool_end_tag,
            hf_tokenizer_path=common_args.tokenizer_path
        )
        
        await test_instance.initialize()
        for t in tasks:
            traj_id = t.get("traj_id", 0)
            await test_instance.run_test_from_gaia_data(
            gaia_sample=t,
            traj_id=traj_id
            )
        
            
        await test_instance.close()

    try:
        asyncio.run(run_test())
        print(f"[Process {worker_id}] Tasks completed.") 
    except Exception as e:
        if worker_id != 0:
            sys.stderr = sys.__stderr__ 
        print(f"[Process {worker_id}] Execution error: {e}")
    finally:
        
        if os.path.exists(worker_input_file):
            os.remove(worker_input_file)
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

def run_single_task_worker(task: Dict[str, Any], common_args: argparse.Namespace):
    """
    This is the *single task* function to be executed by each independent process (worker).
    (Prepared for dynamic queue 'imap_unordered')
    """
    

    worker_id = os.getpid()
    
 
    is_logging_process = (worker_id % common_args.num_processes) == 0

    if not is_logging_process:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')
        import logging
        logger_names = list(logging.Logger.manager.loggerDict.keys()) + [""]
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.handlers = []
            logger.addHandler(logging.NullHandler())
            logger.propagate = False

    print(f"[Process {worker_id}] Starting task: {task.get('task_id')}")


    async def run_test():
        print(f"[DEBUG in run_evaluation.py] 'return_thought' flag value is: {common_args.return_thought}")
        

        test_instance = GaiaApiTest(
            # --- Main Model Parameters ---
            main_provider=common_args.provider,
            main_model=common_args.model,
            main_api_key=common_args.api_key,
            main_base_url=common_args.base_url,
            main_temperature=common_args.temperature,
            main_top_p=common_args.top_p,
            main_presence_penalty=common_args.presence_penalty,
            main_max_tokens=common_args.max_tokens,
            # --- Processor Model Parameters ---
            processor_provider=common_args.processor_provider,
            processor_model=common_args.processor_model,
            processor_api_key=common_args.processor_api_key,
            processor_base_url=common_args.processor_base_url,
            # --- Passing Flags ---
            use_browser_processor=common_args.use_browser_processor,
            return_thought=common_args.return_thought,
            use_context_manager=common_args.use_context_manager,
            # --- General Parameters ---
            manager_url=common_args.manager_url,
            max_interactions=common_args.max_interactions,
            output_dir=str(common_args.output_dir),
            files_dir=common_args.files_dir,
            tool_start_tag=common_args.tool_start_tag,
            tool_end_tag=common_args.tool_end_tag,
            hf_tokenizer_path=common_args.tokenizer_path
        )
        
        await test_instance.initialize()
        

        traj_id = task.get("traj_id", 0)
        await test_instance.run_test_from_gaia_data(
        gaia_sample=task,
        traj_id=traj_id
    )

   
            
        await test_instance.close()


    try:
        asyncio.run(run_test())
        print(f"[Process {worker_id}] Task {task.get('task_id')} completed.")
    except Exception as e:
        if not is_logging_process:
            sys.stderr = sys.__stderr__ 
        print(f"[Process {worker_id}] Task {task.get('task_id')} execution error: {e}")
    


def run_gaia_specific_report(args: argparse.Namespace):

    print("\n" + "="*50)
    print("GAIA Benchmark detected, starting dedicated in-depth analysis report generation...")
    print("="*50)


    report_dir = Path(args.gaia_reports_dir)
    os.makedirs(report_dir, exist_ok=True)
    

    report_base_name = f"{args.model_name}"
    output_file = report_dir / f"{report_base_name}_main_report.md"
    output_text_only_file = report_dir / f"{report_base_name}_text_only_report.md"
    output_llm_judge_file = report_dir / f"{report_base_name}_llm_judge_report.md"
    

    gaia_report_script_path = evaluation_dir / "gaia_report_generator.py"
    if not gaia_report_script_path.exists():
        print(f"\n[Error] Cannot find GAIA report generation script: {gaia_report_script_path}")
 
        gaia_report_script_path_alt = evaluation_dir.parent / "AgentToLeaP" / "gaia_report_generator.py"
        if gaia_report_script_path_alt.exists():
            gaia_report_script_path = gaia_report_script_path_alt
            print(f"Successfully found script in alternative path: {gaia_report_script_path}")
        else:
            print(f"Script not found in alternative path either: {gaia_report_script_path_alt}")
            return


  
    command = [
        sys.executable,
        str(gaia_report_script_path),
        '--result-dir', args.output_dir,
        '--data', args.gaia_metadata_file,
        '--text-data', args.gaia_text_data_file,
        '--output', str(output_file),
        '--output-text-only', str(output_text_only_file),
    ]


    if args.llm_judge:
        command.append('--llm-judge')
        command.extend(['--output-llm-judge', str(output_llm_judge_file)])
        if args.llm_judge_api_key:
            command.extend(['--api-key', args.llm_judge_api_key])
        if args.llm_judge_api_base:
            command.extend(['--api-base', args.llm_judge_api_base])
    
    print("Executing GAIA dedicated report command:")
    
    print(" ".join(f'"{c}"' if " " in c else c for c in command))
    print("-" * 50)

    # Execute command
    try:
        subprocess.run(command, check=True)
        print("\nGAIA dedicated report generated successfully!")
        print(f"Report saved to directory: {report_dir.resolve()}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Failed to execute GAIA dedicated report script: {e}")
    except FileNotFoundError:
        print(f"\n[Error] Cannot find analysis script: {gaia_report_script_path}")


def run_post_evaluation(args: argparse.Namespace):
    """
    Automatically call the evaluate_and_report.py script for analysis and report generation after evaluation is complete.
    """
    print("\n" + "="*50)
    print("Model evaluation complete, now starting result analysis and report generation...")
    print("="*50)

  
    benchmark_name = args.benchmark_name
    

    eval_script_path = evaluation_dir / "evaluate_and_report.py"

    output_base_dir = Path(args.output_base_dir)
    
    # [Modified] Construct model-specific output directory (e.g., evaluation_outputs/gpqa/deepseek)
    # This is where success/fail and LLM Judge reports are stored
    model_specific_dir = output_base_dir / benchmark_name / args.model_name
   
    os.makedirs(model_specific_dir, exist_ok=True)
    
    # SFT files and regular reports are STILL stored in the root directory of output_base_dir
    tasks_output_file = output_base_dir / f"{benchmark_name}_{args.model_name}_correct_tasks.jsonl"
    dialogs_output_file = output_base_dir / f"{benchmark_name}_{args.model_name}_correct_dialogs.json"
    report_output_file = output_base_dir / f"{benchmark_name}_{args.model_name}_report.md"
    
    # [Modified] LLM Judge report is stored in the model-specific directory
    llm_judge_report_file = model_specific_dir / f"{benchmark_name}_{args.model_name}_llm_judge_report.md"


    # Construct command to execute (base part)
    command = [
        sys.executable, 
        str(eval_script_path),
        '--ground-truth-file', args.input_file,
        '--results-dir', args.output_dir,
        '--benchmark-name', benchmark_name,
        '--model-name', args.model_name,
        '--output-base-dir', args.output_base_dir,
        '--tasks-output-file', str(tasks_output_file),
        '--dialogs-output-file', str(dialogs_output_file),
        '--report-output-file', str(report_output_file)
    ]
    
    # --- [New] Dynamically add LLM Judge parameters ---
    if args.llm_judge:
        print("Analysis script will enable [LLM as Judge] mode.")
        command.append('--llm-judge')
        # Pass the newly constructed dedicated report path
        command.extend(['--llm-judge-report-file', str(llm_judge_report_file)])
        
        if args.llm_judge_api_key:
            command.extend(['--llm-judge-api-key', args.llm_judge_api_key])
        # Check if base_url is None or empty string
        if args.llm_judge_api_base: 
            command.extend(['--llm-judge-api-base', args.llm_judge_api_base])

    print("Executing analysis command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in command))
    print("-" * 50)

    # Execute command
    try:
        subprocess.run(command, check=True)
        print("\nResult analysis and report generation successful!")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Failed to execute result analysis script: {e}")
    except FileNotFoundError:
        print(f"\n[Error] Cannot find analysis script: {eval_script_path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation engine for specific benchmarks and automatically generate reports")
    
    # --- Model Related Parameters ---
    parser.add_argument('--provider', type=str, required=True, help="LLM provider (e.g., openai)")
    parser.add_argument('--model', type=str, required=True, help="Model name or path")
    parser.add_argument('--api-key', type=str, help="API Key")
    parser.add_argument('--base-url', type=str, help="API Base URL")
    parser.add_argument('--manager-url', type=str, required=True, help="API URL of MCPManager")
    # --- New Processor Model Parameters ---
    parser.add_argument('--processor-provider', type=str, required=True, help="[Processor Model] LLM provider")
    parser.add_argument('--processor-model', type=str, required=True, help="[Processor Model] Model name or path")
    parser.add_argument('--processor-api-key', type=str, help="[Processor Model] API Key")
    parser.add_argument('--processor-base-url', type=str, help="[Processor Model] API Base URL")
    # --- Name Parameters for Reporting and Classification ---
    parser.add_argument('--model-name', type=str, required=True, help="Short model name used for report generation and folder naming (e.g., deepseek)")

    # --- Data and Output Path Parameters ---
    parser.add_argument('--input-file', type=str, default="input.jsonl", help="Standardized input data file")
    parser.add_argument('--benchmark-name', type=str, required=True, help="Current benchmark name being evaluated (e.g., gpqa)") 
    parser.add_argument('--output-dir', type=str, default="outputs", help="Directory to save [raw] evaluation results")
    parser.add_argument('--files-dir', type=str, default=None, help="Root directory where evaluation files are located (optional)")
    
    # --- Output Path for Final Analysis Results ---
    parser.add_argument('--output-base-dir', type=str, required=True, help="Total base directory for storing all [classified] results and reports (e.g., evaluation_outputs)")

    # --- Evaluation Control Parameters ---
    parser.add_argument('--max-samples', type=int, default=-1, help="Maximum number of evaluation samples (-1 means all)")
    parser.add_argument('--max-interactions', type=int, default=10, help="Maximum number of conversation rounds")
    
    # --- Concurrency Control Parameters ---
    parser.add_argument('--num-processes', type=int, default=1, help="Number of concurrent processes (default is 1, no concurrency)")
    parser.add_argument('--k', type=int, default=1, help="[Pass@k] Number of samples for each task (default is 1, i.e., Pass@1)")
    parser.add_argument('--temperature', type=float, default=1.0, help="[Pass@k] Temperature for model sampling (suggested > 0 for Pass@k, e.g., 0.7)")
    parser.add_argument('--top-p', type=float, default=1.0, help="Top P for model sampling")
    parser.add_argument('--presence-penalty', type=float, default=1.0, help="Presence Penalty for model sampling")
    parser.add_argument('--max-tokens', type=int, default=16384, help="Maximum number of tokens generated by the model")
    parser.add_argument('--tool-start-tag', type=str, default=None, help="Start tag for tool calls (optional)")
    parser.add_argument('--tool-end-tag', type=str, default=None, help="End tag for tool calls (optional)")
    parser.add_argument('--tokenizer-path', type=str, default=None, help="HuggingFace tokenizer path or model name")

    parser.add_argument(
        '--use-browser-processor', 
        action='store_true', 
        help="If this flag is provided, the Browser Processor Agent is enabled to summarize and sanitize browser content"
    )

    parser.add_argument(
        '--return-thought',
        action='store_true',
        help="If this flag is provided, the previous round's thought process will be injected into the assistant's content and returned to the LLM"
    )
    
    parser.add_argument(
        '--use-context-manager',
        action='store_true',
        help="If this flag is provided, the HistoryX context manager (history summarization) function is enabled"
    )

    parser.add_argument(
        '--llm-judge',
        action='store_true',
        help="[New] Enable LLM as Judge for scoring, which will override the 'four-layer funnel' logic."
    )
    parser.add_argument(
        '--llm-judge-api-key', 
        type=str, 
        help="[New] API Key used by LLM Judge."
    )
    parser.add_argument(
        '--llm-judge-api-base', 
        type=str, 
        default="https://api.openai.com/v1", 
        help="[New] API Base URL used by LLM Judge."
    )

    parser.add_argument(
        '--skip-post-eval',
        action='store_true',
        help="[Main Control Script] If this flag is provided, the built-in 'evaluate_and_report.py' analysis step is skipped"
    )
    parser.add_argument('--run-gaia-specific-report', type=str, default='false', help="Whether to run the dedicated report generation script for the GAIA benchmark (pass 'true' or 'false')")
    parser.add_argument('--gaia-metadata-file', type=str, help="[GAIA Dedicated] GAIA metadata file path (.jsonl)")
    parser.add_argument('--gaia-text-data-file', type=str, help="[GAIA Dedicated] GAIA text task data file path (.json)")
    parser.add_argument('--gaia-reports-dir', type=str, help="[GAIA Dedicated] Total directory for storing all GAIA reports")



    args = parser.parse_args()
    
    print(f"--- Starting Benchmark Evaluation ---")
    print(f"Model: {args.model}")
    print(f"Tool Server: {args.manager_url}")
    print(f"Data Source: {args.input_file}")
    print(f"Concurrency: {args.num_processes}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_tasks_original = [json.loads(line) for line in f if line.strip()]
        
        if args.max_samples != -1:
            all_tasks_original = all_tasks_original[:args.max_samples]
        
        print(f"Loaded a total of {len(all_tasks_original)} original tasks.")

 
        all_tasks = []
        if args.k > 1:
            if args.temperature == 0 or args.temperature == 0.0:
                print(f"[Warning] Pass@{args.k} mode enabled, but temperature is still 0. The output of all run instances might be identical.")
            print(f"Running in Pass@{args.k} mode (Temperature={args.temperature}). Expanding tasks...")
            # Benefit: The first run of all tasks will be completed first, then the second run, etc., making it easier to view partial results midway.
            for i in range(args.k):
                for task in all_tasks_original:
                    original_task_id = task.get("task_id")
                    if original_task_id is None:
                        print(f"[Error] Task {task} is missing 'task_id', cannot perform Pass@k expansion.")
                        continue
                    new_task = json.loads(json.dumps(task)) # Deep copy
                    new_task["task_id"] = original_task_id 
                    new_task["traj_id"] = i  
                    new_task["run_id"] = i  
                    all_tasks.append(new_task)
        else:
            all_tasks = all_tasks_original
        
        print(f"Total number of tasks to process (total {len(all_tasks)} run instances): {len(all_tasks)}")
        

    except FileNotFoundError:
        print(f"[Error] Input file not found: {args.input_file}")
        return

    if args.num_processes <= 1:
        print("Running in single-process mode...")
        run_worker_task(all_tasks, 0, args)
    else:
        print(f"Running in [Dynamic Queue] mode, starting {args.num_processes} processes...")
        
        # 1. Use partial (partial function) to "fix" the common_args parameter
        #    This way, the worker function only needs one parameter: task
        worker_func = partial(run_single_task_worker, common_args=args)

        # 2. Start process pool
        with Pool(processes=args.num_processes) as pool:
            
            # 3. Use imap_unordered to implement a dynamic task queue
            #    - It takes tasks from the 'all_tasks' list
            #    - 'chunksize=1' ensures that each process takes only 1 task at a time
            #    - After a process completes a task, it automatically returns to take the next one, achieving "asynchrony"
            print(f"Starting execution of {len(all_tasks)} tasks...")
            
            # Use list() to "exhaust" the iterator, ensuring all tasks are completed
            try:
                list(pool.imap_unordered(worker_func, all_tasks, chunksize=1))
            except Exception as e:
                print(f"[Critical Error] Process pool execution failed: {e}")
                pool.terminate() # Terminate all child processes when an error occurs

    print(f"--- Benchmark evaluation fully completed ---")
    print(f"All raw results saved in: {args.output_dir}")
    
    is_gaia_report_enabled = getattr(args, 'run_gaia_specific_report', 'false').lower() == 'true'

    if args.skip_post_eval:
        # If the main switch skipped, do nothing
        print("\n[Control] Evaluation tasks completed, built-in post-processing analysis skipped.")
    
    elif (args.benchmark_name == 'gaia' or args.benchmark_name =='gaia_text') and is_gaia_report_enabled:
        # If it's a GAIA task and dedicated report generation is enabled, run ONLY the GAIA dedicated script
        print("\n[Control] GAIA benchmark detected, will run the GAIA dedicated report script...")
        run_gaia_specific_report(args)
        
    else:
        # For all other benchmarks (or when GAIA dedicated report is not enabled), run the general script
        print(f"\n[Control] Benchmark '{args.benchmark_name}' detected, will run the general evaluation script...")
        run_post_evaluation(args)

if __name__ == "__main__":
    main()
