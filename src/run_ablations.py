"""
Script for running systematic ablation studies.

Executes the reproduction pipeline (`tool.py` or `tool_openai.py`) with various
configuration flags to test the impact of different components (retrieval, generation phases).
"""

import subprocess
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List
from repgen_logging import setup_logging, trace

# --- 1. Define Retrieval Ablations ---
# These map to the --retrieval_ablation flag in the tool script
RETRIEVAL_ABLATIONS = [
    # "full_system",
    "NO_BM25",
    "NO_ANN",
    "NO_RERANKER",
    "NO_TRAINING_LOOP_EXTRACTION",
    "NO_TRAINING_LOOP_RANKING",
    "NO_MODULE_PARTITIONING",
    "NO_DEPENDENCY_EXTRACTION",
]

# --- 2. Define Generation Ablations ---
# These map to the --generation_ablation flag in the tool script
GENERATION_ABLATIONS = [
    "all_steps",
    "no_refine",
    "no_plan",
    "no_compilation",
    "no_relevance",
    "no_static_analysis",
    "no_runtime_feedback", 
]

# --- 3. Setup Logging ---
script_logger = setup_logging(logging.INFO).bind(component="runner.ablations")

# --- 4. Retry Function ---
def run_command_with_retry(cmd: List[str], log_file: Path, max_attempts: int = 3):
    """
    Runs a command with retries and logs stdout/stderr to a file.

    Args:
        cmd: List of command-line arguments.
        log_file: Path to the log file.
        max_attempts: Maximum number of retry attempts.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    for attempt in range(1, max_attempts + 1):
        trace(script_logger, "Launching ablation command", stage="runner", action="execute_command", status="running", attempt=attempt, details={"max_attempts": max_attempts, "command": " ".join(cmd), "log_file": str(log_file)})
        
        try:
            # Capture both stdout and stderr to the log file
            with open(log_file, 'w', encoding='utf-8') as f:
                process = subprocess.run(
                    cmd,
                    check=True, 
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            trace(script_logger, "Ablation command succeeded", stage="runner", action="execute_command", status="ok", attempt=attempt, details={"log_file": str(log_file)})
            return 0 # Success
        except subprocess.CalledProcessError as e:
            trace(script_logger, "Ablation command failed; retrying", level=logging.WARNING, stage="runner", action="execute_command", status="retry", attempt=attempt, details={"exit_code": e.returncode, "log_file": str(log_file)})
            time.sleep(2)
        except FileNotFoundError:
            trace(script_logger, "Python executable or target script missing", level=logging.ERROR, stage="runner", action="execute_command", status="error")
            return 127 # Command not found
        except Exception as e:
            trace(script_logger, f"Unexpected runner error: {e}", level=logging.ERROR, stage="runner", action="execute_command", status="retry", attempt=attempt)
            time.sleep(2)
    
    trace(script_logger, "Ablation command failed after max retries", level=logging.ERROR, stage="runner", action="execute_command", status="error", details={"max_attempts": max_attempts, "log_file": str(log_file)})
    return 1 # Failure

# --- 5. Main Execution Logic ---
def main():
    """
    Main execution function for ablation studies.

    Parses arguments for bug ID range, tool script, and retry settings.
    Iterates through bugs and ablation configurations (Retrieval, Generation),
    executing the pipeline and logging results.
    """
    parser = argparse.ArgumentParser(description="Systematic ablation runner for RepGen")
    parser.add_argument("--start_bug_id", required=True, type=int, help="First bug ID to process (e.g., 1)")
    parser.add_argument("--end_bug_id", required=True, type=int, help="Last bug ID to process (e.g., 10)")
    
    # New configuration arguments
    parser.add_argument("--tool_script", type=str, default="tool_openai.py", 
                        help="The script to run (e.g., tool.py or tool_openai.py)")
    parser.add_argument("--max-gen-attempts", type=int, default=5, 
                        help="Max attempts for *each* code generation run (passed to tool)")
    parser.add_argument("--max-run-attempts", type=int, default=3,
                        help="Max retry attempts for *each script execution*")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional custom dataset path to pass to the tool")

    args = parser.parse_args()

    if args.start_bug_id > args.end_bug_id:
        script_logger.error("Error: Start bug ID must be less than or equal to end bug ID")
        sys.exit(1)

    # Verify the tool script exists
    if not os.path.exists(args.tool_script):
        script_logger.error(f"Error: Tool script '{args.tool_script}' not found.")
        sys.exit(1)

    base_dir = Path(__file__).parent
    logs_base_dir = base_dir / "logs"

    # Loop through bug IDs
    for i in range(args.start_bug_id, args.end_bug_id + 1):
        bug_id = f"{i:03d}" # Format as "001", "002", etc.
        script_logger.info(f"\n===== Processing bug_id: {bug_id} using {args.tool_script} =====")
        
        logs_dir = logs_base_dir / f"bug_{bug_id}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # --- STUDY 1: Retrieval Ablations (ACTIVE) ---
        script_logger.info("========== RUNNING STUDY 1: RETRIEVAL ABLATIONS ==========")
        
        for ret_ablation in RETRIEVAL_ABLATIONS:
            if ret_ablation == "full_system":
                log_name = f"bug_{bug_id}_baseline_retrieval.log"
                script_logger.info("\nRunning baseline retrieval (full_system):")
            else:
                log_name = f"bug_{bug_id}_{ret_ablation}.log"
                script_logger.info(f"\nRunning with {ret_ablation}:")
                
            log_file_path = logs_dir / log_name

            # Construct command dynamically
            cmd = [
                "python", args.tool_script,
                f"--bug_id={bug_id}",
                f"--max-attempts={args.max_gen_attempts}",
                f"--retrieval_ablation={ret_ablation}",
                "--generation_ablation=all_steps" # Always use full generation pipeline for retrieval study
            ]
            
            # Pass dataset path if provided (assuming the tool script accepts --ae_dataset_path)
            if args.dataset_path:
                cmd.append(f"--ae_dataset_path={args.dataset_path}")
            
            run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

        script_logger.info("========== RUNNING STUDY 2: GENERATION ABLATIONS ==========")
        # We skip the first item ("all_steps") because it was run in Study 1 (as full_system)
        for gen_ablation in GENERATION_ABLATIONS[1:]:
            script_logger.info(f"\nRunning with {gen_ablation}:")
            log_name = f"bug_{bug_id}_{gen_ablation}.log"
            log_file_path = logs_dir / log_name
            
            cmd = [
                "python", args.tool_script,
                f"--bug_id={bug_id}",
                f"--max-attempts={args.max_gen_attempts}",
                "--retrieval_ablation=full_system", 
                f"--generation_ablation={gen_ablation}"
            ]
            
            if args.dataset_path:
                cmd.append(f"--ae_dataset_path={args.dataset_path}")

            run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

    script_logger.info(f"\n===== Completed processing all bug IDs from {args.start_bug_id} to {args.end_bug_id} =====")

if __name__ == "__main__":
    main()
