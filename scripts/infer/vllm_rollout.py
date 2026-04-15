import argparse
import pandas as pd
import json
import os
import multiprocessing
from collections import Counter


DEFAULT_GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
DEFAULT_ENABLE_THINKING = False
DEFAULT_ENABLE_REJECTION_SAMPLING = True
DEFAULT_MAX_ATTEMPTS_PER_ROLLOUT = 3


# ───────────────────────── Rejection Sampling Filters ─────────────────────────

def has_boxed(text: str) -> bool:
    """Check if output contains \\boxed{}."""
    return "\\boxed" in text


def detect_repeated_lines(text: str, min_len: int = 20, threshold: int = 5) -> bool:
    """Detect if any line (len >= min_len) appears >= threshold times."""
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= min_len]
    if not lines:
        return False
    counter = Counter(lines)
    return counter.most_common(1)[0][1] >= threshold


def detect_ngram_repetition(text: str, n: int = 100, threshold: int = 3) -> bool:
    """Detect if any n-char substring appears >= threshold times (sliding window)."""
    if len(text) < n * threshold:
        return False
    seen = {}
    for i in range(0, len(text) - n + 1, 10):
        chunk = text[i : i + n]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


def detect_consecutive_repeat(text: str, block_size: int = 50, threshold: int = 3) -> bool:
    """Detect if a consecutive block of text (>= block_size) repeats >= threshold times in a row."""
    if len(text) < block_size * threshold:
        return False
    for i in range(len(text) - block_size * threshold + 1):
        block = text[i : i + block_size]
        count = 1
        pos = i + block_size
        while pos + block_size <= len(text) and text[pos : pos + block_size] == block:
            count += 1
            pos += block_size
            if count >= threshold:
                return True
    return False


def is_valid_output(text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Reject outputs that are missing \\boxed{} or contain repetitive text.
    """
    if not has_boxed(text):
        return False, "no_boxed"
    if detect_repeated_lines(text):
        return False, "repeated_lines"
    if detect_ngram_repetition(text):
        return False, "ngram_repetition"
    if len(text) > 5000 and detect_consecutive_repeat(text):
        return False, "consecutive_repeat"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────

def extract_instruction(raw_chat):
    if isinstance(raw_chat, (list, bytes, pd.Series)) or hasattr(raw_chat, "__getitem__"):
        try:
            return raw_chat[0]["content"]
        except Exception:
            return str(raw_chat)
    return str(raw_chat)


def parse_bool(value: str) -> bool:
    """Parse explicit boolean CLI values such as true/false or 1/0."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_gpu_ids(value: str) -> list[int]:
    """Parse a comma-separated GPU ID list like 0,1,2,3."""
    gpu_ids = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            gpu_id = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid GPU ID: {part}") from exc
        if gpu_id < 0:
            raise argparse.ArgumentTypeError(f"GPU IDs must be non-negative integers: {part}")
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise argparse.ArgumentTypeError("At least one GPU ID must be provided.")
    return gpu_ids


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for rollout configuration."""
    parser = argparse.ArgumentParser(
        description="Roll out teacher responses for subsequent student SFT."
    )
    parser.add_argument(
        "--input-parquet",
        required=True,
        help="Path to the parquet file containing the prompt column.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the teacher model used for rollout generation.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=DEFAULT_GPU_IDS.copy(),
        help="Comma-separated GPU IDs to use, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=parse_bool,
        default=DEFAULT_ENABLE_THINKING,
        help="Whether to enable thinking / reasoning templates for compatible models.",
    )
    parser.add_argument(
        "--enable-rejection-sampling",
        type=parse_bool,
        default=DEFAULT_ENABLE_REJECTION_SAMPLING,
        help="Whether to reject invalid generations and retry rollout slots.",
    )
    parser.add_argument(
        "--max-attempts-per-rollout",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_ROLLOUT,
        help="Maximum retry count for each rollout slot when rejection sampling is enabled.",
    )
    return parser


def worker(
    rank: int,
    gpu_ids: list[int],
    data_slice: list,          # Original prompt list handled by this worker (list of chats)
    global_indices: list[int], # Corresponding global indices used for result alignment
    model_path: str,
    temp_jsonl: str,           # Worker-specific checkpoint file
    sampling_params_kwargs: dict,
    num_rollouts: int = 4,
    enable_thinking: bool = False,
    enable_rejection_sampling: bool = True,
    max_attempts_per_rollout: int = 10,  # Maximum retry count for each rollout slot
):
    """A single worker process that owns its GPU and streams rollout outputs to disk."""

    # ---------- Must be set before importing vllm, or NCCL may grab all GPUs ----------
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    port_offset = 29500 + rank * 10
    os.environ["VLLM_PORT"] = str(port_offset)
    os.environ["MASTER_PORT"] = str(port_offset)
    os.environ["NCCL_PORT"] = str(port_offset)

    from vllm import LLM, SamplingParams  # Delayed import so env vars are already in effect
    import gc
    import torch

    if enable_rejection_sampling:
        mode_desc = f"rejection sampling enabled, up to {max_attempts_per_rollout} retries per slot"
    else:
        mode_desc = "rejection sampling disabled, all generations will be saved directly"
    print(f"[Worker {rank}] Using GPU {gpu_ids} for {len(data_slice)} samples, {mode_desc}")

    # -------- Checkpoint resume: count completed valid rollouts for each sample --------
    completed_rollouts = {}
    if os.path.exists(temp_jsonl):
        with open(temp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    gidx = record["global_index"]
                    rollout_index = record.get("rollout_index")
                    if isinstance(rollout_index, int) and 0 <= rollout_index < num_rollouts:
                        completed_rollouts.setdefault(gidx, set()).add(rollout_index)
                except Exception:
                    continue
        finished_prompts = sum(
            1 for rollout_set in completed_rollouts.values() if len(rollout_set) >= num_rollouts
        )
        print(f"[Worker {rank}] Checkpoint detected, {finished_prompts} prompts already have completed rollouts")

    # Build the initial pending pool and fill missing rollout slots for each prompt.
    # Each item also carries an attempts field to track retries for the current slot.
    pending_pool = []
    for i in range(len(data_slice)):
        gidx = global_indices[i]
        done_rollout_indices = completed_rollouts.get(gidx, set())
        for rollout_index in range(num_rollouts):
            if rollout_index in done_rollout_indices:
                continue
            pending_pool.append({
                "gidx": gidx,
                "chat": data_slice[i],
                "rollout_index": rollout_index,
                "attempts": 0,
            })

    if not pending_pool:
        print(f"[Worker {rank}] All assigned data has already been processed.")
        return

    # -------- Initialize model --------
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=len(gpu_ids),
            max_model_len=10480,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        sampling_params = SamplingParams(**sampling_params_kwargs)
        tokenizer = llm.get_tokenizer()

        # -------- Generate rollouts and write them to disk --------
        batch_size = 1024
        total_slots = len(pending_pool)   # Total number of slots that still need outputs
        saved_count = 0
        reject_stats = {"no_boxed": 0, "repeated_lines": 0, "ngram_repetition": 0, "consecutive_repeat": 0}
        skipped_count = 0                 # Slots abandoned after exceeding max retries

        with open(temp_jsonl, "a", encoding="utf-8") as f_out:
            while pending_pool:
                current_batch = pending_pool[:batch_size]
                pending_pool = pending_pool[batch_size:]

                # Format prompts
                formatted_prompts = [
                    tokenizer.apply_chat_template(
                        item["chat"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    for item in current_batch
                ]

                # Generate
                outputs = llm.generate(formatted_prompts, sampling_params)

                retry_items = []  # Items rejected in this round and queued again

                for item, output in zip(current_batch, outputs):
                    model_output = output.outputs[0].text
                    gidx = item["gidx"]
                    rollout_index = item["rollout_index"]
                    if enable_rejection_sampling:
                        attempts = item["attempts"]
                        valid, reason = is_valid_output(model_output)

                        if not valid:
                            reject_stats[reason] = reject_stats.get(reason, 0) + 1
                            if attempts + 1 < max_attempts_per_rollout:
                                retry_items.append({
                                    "gidx": gidx,
                                    "chat": item["chat"],
                                    "rollout_index": rollout_index,
                                    "attempts": attempts + 1,
                                })
                            else:
                                skipped_count += 1
                                print(
                                    f"[Worker {rank}] ⚠ gidx={gidx} rollout_slot={rollout_index} "
                                    f"reached the retry limit of {max_attempts_per_rollout}; abandoning this slot (last rejection reason: {reason})"
                                )
                            continue

                    # Valid output or direct-save mode -> write to disk
                    record = {
                        "global_index": gidx,
                        "rollout_index": rollout_index,
                        "instruction": extract_instruction(item["chat"]),
                        "input": "",
                        "output": model_output,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    saved_count += 1

                # Append retry items to the end of the queue
                pending_pool.extend(retry_items)

                if enable_rejection_sampling:
                    total_rejected = sum(reject_stats.values())
                    print(
                        f"[Worker {rank}] Progress: accepted {saved_count}/{total_slots} slots | "
                        f"pending: {len(pending_pool)} | "
                        f"rejected: {total_rejected} (no_boxed={reject_stats['no_boxed']}, "
                        f"rep_lines={reject_stats['repeated_lines']}, "
                        f"ngram={reject_stats['ngram_repetition']}, "
                        f"consec={reject_stats['consecutive_repeat']}) | "
                        f"abandoned: {skipped_count}"
                    )
                else:
                    print(
                        f"[Worker {rank}] Progress: saved {saved_count}/{total_slots} slots | "
                        f"pending: {len(pending_pool)} | rejection sampling disabled"
                    )

        if enable_rejection_sampling:
            total_rejected = sum(reject_stats.values())
            print(f"[Worker {rank}] Inference finished. Saved {saved_count} valid rollouts, "
                  f"rejected {total_rejected} times in total, and abandoned {skipped_count} slots")
        else:
            print(f"[Worker {rank}] Inference finished. Rejection sampling was disabled, so {saved_count} rollouts were saved directly")

    finally:
        # -------- Explicitly clean up vLLM processes and GPU memory --------
        print(f"[Worker {rank}] Cleaning up vLLM resources...")
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            print(f"[Worker {rank}] Error while cleaning up the distributed environment (ignored): {e}")

        if 'llm' in locals():
            del llm
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Worker {rank}] Resource cleanup completed.")


def main():
    args = build_arg_parser().parse_args()

    # -------------------- Path configuration --------------------
    input_parquet = args.input_parquet
    model_path = args.model_path
    model_name     = os.path.basename(model_path.rstrip("/"))
    base_dir       = f"output/{model_name}"
    output_jsonl   = os.path.join(base_dir, "DAPO.jsonl")
    temp_dir       = os.path.join(base_dir, "temp_rollout")
    os.makedirs(base_dir, exist_ok=True)  # Ensure the model-named output directory exists

    # GPU IDs used by each worker (one instance per GPU)
    gpu_ids_all = args.gpu_ids
    num_workers = len(gpu_ids_all)
    gpu_groups = [[gid] for gid in gpu_ids_all]

    enable_thinking = args.enable_thinking
    enable_rejection_sampling = args.enable_rejection_sampling
    max_attempts_per_rollout = args.max_attempts_per_rollout

    sampling_params_kwargs = dict(
        temperature=1.0,
        max_tokens=7168,
        top_k=-1,
        top_p=0.95
    )

    os.makedirs(temp_dir, exist_ok=True)

    # -------------------- Load data --------------------
    if not os.path.exists(input_parquet):
        print(f"Error: {input_parquet} does not exist.")
        return

    print(f"Loading data: {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    prompts_raw = df["prompt"].tolist()
    total_data  = len(prompts_raw)
    print(f"Loaded {total_data} samples and assigned them to {num_workers} workers.")

    num_rollouts = 1

    # -------------------- Global checkpoint statistics --------------------
    # Collect completed rollouts from all worker checkpoint files
    global_completed_rollouts = {}
    for rank in range(num_workers):
        temp_jsonl = os.path.join(temp_dir, f"worker_{rank}.jsonl")
        if not os.path.exists(temp_jsonl):
            continue
        with open(temp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    gidx = record.get("global_index")
                    rollout_index = record.get("rollout_index")
                    if gidx is None:
                        continue
                    if isinstance(rollout_index, int) and 0 <= rollout_index < num_rollouts:
                        global_completed_rollouts.setdefault(gidx, set()).add(rollout_index)
                except Exception:
                    continue

    completed_prompt_count = sum(
        1 for rollout_set in global_completed_rollouts.values() if len(rollout_set) >= num_rollouts
    )
    completed_rollout_count = sum(
        min(len(rollout_set), num_rollouts) for rollout_set in global_completed_rollouts.values()
    )
    remaining_count = total_data - completed_prompt_count
    print(
        f"Checkpoint status: {completed_prompt_count} prompts completed | "
        f"{completed_rollout_count} rollouts completed | {remaining_count} prompts remaining"
    )

    # -------------------- Data sharding --------------------
    # Re-shard only the remaining samples so workers do actual inference instead of only skipping entries
    remaining_samples = [
        (gidx, chat)
        for gidx, chat in enumerate(prompts_raw)
        if len(global_completed_rollouts.get(gidx, set())) < num_rollouts
    ]

    need_run_workers = len(remaining_samples) > 0
    if not need_run_workers:
        print("All samples are already complete in the checkpoint files; skipping multiprocess resume.")

    chunks      = [[] for _ in range(num_workers)]
    idx_chunks  = [[] for _ in range(num_workers)]
    for i, (gidx, chat) in enumerate(remaining_samples):
        wid = i % num_workers
        chunks[wid].append(chat)
        idx_chunks[wid].append(gidx)

    # -------------------- Launch multiprocessing --------------------
    if need_run_workers:
        # Use spawn to avoid CUDA issues under fork mode
        ctx = multiprocessing.get_context("spawn")
        processes = []

        for rank in range(num_workers):
            temp_jsonl = os.path.join(temp_dir, f"worker_{rank}.jsonl")
            p = ctx.Process(
                target=worker,
                kwargs=dict(
                    rank=rank,
                    gpu_ids=gpu_groups[rank],
                    data_slice=chunks[rank],
                    global_indices=idx_chunks[rank],
                    model_path=model_path,
                    temp_jsonl=temp_jsonl,
                    sampling_params_kwargs=sampling_params_kwargs,
                    num_rollouts=num_rollouts,
                    enable_thinking=enable_thinking,
                    enable_rejection_sampling=enable_rejection_sampling,
                    max_attempts_per_rollout=max_attempts_per_rollout,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check whether any worker exited unexpectedly
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"Warning: Worker {failed} exited unexpectedly. Please check the logs and rerun; checkpoint resume will continue automatically.")
            return
        print("All workers finished the resumed run. Merging outputs and computing statistics...")
    else:
        print("No workers were launched. Merging existing checkpoint results directly...")

    # -------------------- Merge outputs and compute statistics --------------------
    all_records = []
    
    for rank in range(num_workers):
        temp_jsonl = os.path.join(temp_dir, f"worker_{rank}.jsonl")
        if not os.path.exists(temp_jsonl):
            continue
        with open(temp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                rollout_index = record.get("rollout_index")
                if isinstance(rollout_index, int) and 0 <= rollout_index < num_rollouts:
                    all_records.append(record)

    # Restore order by global_index and rollout_index
    all_records.sort(key=lambda x: (x["global_index"], x.get("rollout_index", 0)))

    # Convert to ShareGPT/OpenAI format
    final_results = []
    for r in all_records:
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": r["instruction"]
                },
                {
                    "role": "assistant",
                    "content": r["output"]
                }
            ]
        }
        final_results.append(record)

    with open(output_jsonl, "w", encoding="utf-8") as f_final:
        for record in final_results:
            f_final.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary statistics
    print("\n" + "="*40)
    print("Rollout Summary")
    print("="*40)
    print(f"Total input samples:       {total_data}")
    print(f"Rollouts per sample:       {num_rollouts}")
    print(f"Total rollouts generated:  {len(final_results)}")
    expected_total = total_data * num_rollouts
    print(f"Expected total rollouts:   {expected_total}")
    print(f"Completion rate:           {len(final_results) / expected_total:.2%}" if expected_total > 0 else "0%")
    print("="*40)
    print(f"Final results saved to: {output_jsonl}")
    
    # Uncomment the line below if you want to remove temporary files:
    # import shutil; shutil.rmtree(temp_dir)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
