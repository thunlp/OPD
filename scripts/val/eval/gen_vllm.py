import os
import json
import random
import re
import argparse
import concurrent.futures
import multiprocessing  # Added for spawn-based worker management
import gc  # Added for explicit resource cleanup
import torch  # Added for CUDA cache cleanup
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
# Try to import distributed cleanup helpers for releasing GPU memory.
try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR = "../data"
MODEL_FOLDER = "../../model/Qwen3-1.7B-SFT-DAPO-4B-filtered"

def extract_max_number(path):
    """Extract all numbers from a path and return the largest one for sorting."""
    numbers = re.findall(r'\d+', path)
    if numbers:
        return max(int(n) for n in numbers)
    return -1  # If there is no number, keep this entry at the end.

# Collect model paths and sort them by descending numeric suffix.
try:
    model_files = os.listdir(MODEL_FOLDER)
    MODEL_NAMES_CANDIDATES = [os.path.join(MODEL_FOLDER, f) for f in model_files]
    MODEL_NAMES_CANDIDATES.sort(key=extract_max_number, reverse=True)
except FileNotFoundError:
    MODEL_NAMES_CANDIDATES = []

# Active model list.
MODEL_NAMES = MODEL_NAMES_CANDIDATES
MODEL_NAMES = ["../../model/Qwen3-4B"]

TASKS = [
    # {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 16},
    # {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 16},
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 16},
]

PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
MAX_TOKENS  = 31744
TEMPERATURE = 0.7
TOP_P       = 0.95
REPLACE     = False

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_rollout_ids(rollout_ids: list[int], num_workers: int):
    """Round-robin split of rollout IDs into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


# --------------------------------------------------------------------------- #
#              Worker process (one model instance per GPU worker)              #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:
    args_tuple = (model_name, samples, rollout_id_list, gpu_id, enable_thinking)
    gpu_id: values such as "0" or "3", used for CUDA_VISIBLE_DEVICES
    """
    model_name, samples, rollout_id_list, gpu_id, enable_thinking = args_tuple
    
    # CUDA_VISIBLE_DEVICES must be set inside the spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    results = []
    llm = None
    stop_token_ids = []
    
    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | rollouts len={len(rollout_id_list)} "
            f"| loading model (TP=1, enable_thinking={enable_thinking})...",
            flush=True,
        )
        
        # Initialize a single-GPU, single-instance LLM.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        
        # Get the tokenizer.
        try:
            tokenizer = llm.get_tokenizer()
            
            # Encode stop tokens.
            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    if hasattr(tokenizer, "encode"):
                        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                        if encoded:
                            stop_token_ids.append(encoded[0])
                except Exception:
                    pass
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer for stop tokens: {e}", flush=True)
        
        for rollout_id in rollout_id_list:
            sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
            )

            if tokenizer is None:
                raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")

            # Do not set request-level seeds here. Per-request generators can
            # force vLLM to fall back from the FlashInfer sampler path.
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for s in samples
            ]
            
            # Disable per-worker tqdm output to keep multi-process logs readable.
            outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
            
            for sample, out in zip(samples, outputs):
                results.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": rollout_id,
                        "response": out.outputs[0].text,
                    }
                )
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)
        # For debugging, error details could be stored in results or logged directly.
    
    finally:
        # Explicitly release vLLM resources.
        # This helps prevent CUDA context deadlocks and zombie processes.
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm
        
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass

        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)

    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation rollouts with vLLM.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking when applying the chat template.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking when applying the chat template.",
    )
    parser.set_defaults(enable_thinking=False)
    args = parser.parse_args()

    # Specify GPU IDs, with one model instance assigned to each GPU.
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_workers = [str(gpu_id) for gpu_id in available_gpus]
    num_workers = len(gpu_workers)

    print(f"GPU workers (one model per GPU): {gpu_workers}")
    print(f"apply_chat_template enable_thinking={args.enable_thinking}")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        
        OUT_DIR = Path(f"justrl_eval_outputs/{model_name.split('/')[-1]}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            task_name = task["name"]
            task_path = task["path"]
            N = task["N"]

            print(f"Starting evaluation for task: {task_name} (N={N})")
            
            out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"

            # --- Repetition Check ---
            if not REPLACE and out_path.exists():
                print(f"Result file already exists at '{out_path}'. Skipping.")
                continue  # Skip to the next task

            # 1. Load original prompts
            samples = load_samples(task_path)

            # Append suffix prompt to each sample
            for sample in samples:
                # Ensure the prompt format is correct.
                sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

            if len(samples) > 0:
                print("Example prompt after formatting:")
                print(samples[0]["prompt"])
            
            # 2. Generate rollout IDs and split across GPUs.
            # These IDs are bookkeeping only; they no longer control vLLM RNG.
            rollout_ids = list(range(N))
            rollout_chunks = split_rollout_ids(rollout_ids, num_workers)

            # 3. Launch workers, with each worker using one GPU.
            all_results = []
            args_list = [
                (model_name, samples, rollout_chunks[i], gpu_workers[i], args.enable_thinking)
                for i in range(num_workers)
            ]
            
            # Use the spawn start method for worker processes.
            ctx = multiprocessing.get_context("spawn")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                
                # Track overall progress with tqdm.
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    try:
                        res = fut.result()
                        all_results.extend(res)
                    except Exception as e:
                        print(f"A worker process failed with error: {e}")

            print(f"Total generations collected for {task_name}: {len(all_results)}")

            # 4. Save to disk
            if all_results:
                with out_path.open("w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Saved results for {task_name} to {out_path}")
            else:
                print(f"No results collected for {task_name} (Check for errors).")


if __name__ == "__main__":
    # Set the start method to avoid multiprocessing issues in some environments.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
