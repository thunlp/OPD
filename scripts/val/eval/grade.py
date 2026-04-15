from utils import grade_answer_verl
from transformers import AutoTokenizer
import json
import pandas as pd
from pathlib import Path
import re
import argparse # Added
# vllm import moved to conditional execution to save resources if disabled

CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

NAME     = "Qwen3-4B-Non-Thinking-RL-Math" 
EVAL_DIR = Path(f"justrl_eval_outputs/{NAME}")
OUTPUT_FILE = EVAL_DIR / "grading_results.json"
MODEL_NAME = "../../model/CompassVerifier-3B"

# Global variables to be initialized if needed
vllm_model = None
model_tokenizer = None
sampling_params = None
length_tokenizer = AutoTokenizer.from_pretrained("../../model/Qwen3-1.7B", local_files_only=True)

def get_len(seq):
    if length_tokenizer:
        return len(length_tokenizer.encode(seq))
    return len(seq)

def get_diverse_score(sequences, n=4):
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0

def process_jsonl_file(file_name):
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            id = int(data["example_id"])
            while len(results) <= id:
                results.append({"gt": None, "responses": []})
            gt = data["answer"]
            response = data["response"]
            results[id]["gt"] = gt
            results[id]["responses"].append(response)
    return results

def parse_hyperparameters_from_filename(filename):
    match = re.search(r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)", filename)
    return match.groupdict() if match else {}

def grade_file(file_path, use_model_verifier=True):
    """
    Grade a single file.
    use_model_verifier: Boolean, if False, only use rule-based matching.
    """
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None

    task_name = file_path.stem.split("_")[0]
    hyperparams["task_name"] = task_name

    if "parquet" in str(file_path):
        df = pd.read_parquet(file_path)
        num_pred = len(df["responses"][0])
    else:
        df = process_jsonl_file(file_path)
        num_pred = len(df[0]["responses"])

    results = {
        "hyperparameters": hyperparams,
        "mean_score": 0,
        "distinct_4gram": 0,
        "best_score": 0,
        "solve_none": 0,
        "solve_all": 0,
        "avg_output_length": 0,
        "format_error_rollouts": 0,
    }

    diverse = []
    avg_scores = []
    best = []
    without_boxed = 0
    response_lengths = []

    all_model_inputs = []
    all_responses = []
    all_questions = []
    all_ground_truths = []
    rule_based_scores = []

    for i in range(len(df)):
        if "jsonl" in str(file_path):
            responses = df[i]["responses"]
            gt = df[i]["gt"]
            question = df[i].get("question", "")
        else:
            responses = df["responses"][i]
            gt = df["reward_model"][i]["ground_truth"]
            question = df["reward_model"][i].get("question", "")

        responses_list = [str(response) for response in responses]
        response_lengths += [get_len(response) for response in responses_list]
        
        not_formated = ["boxed" not in response for response in responses_list]
        without_boxed += sum(not_formated)

        for response in responses_list:
            rule_score = grade_answer_verl(response, gt)
            rule_based_scores.append(rule_score)
            
            # Only prepare for model verification if rule-based failed AND verification is enabled
            if not rule_score and use_model_verifier:
                model_input = CV_PROMPT.format(
                    question=question,
                    gold_answer=gt,
                    llm_response=response
                )
                all_model_inputs.append(model_input)
                all_responses.append(response)
                all_questions.append(question)
                all_ground_truths.append(gt)

        diverse.append(get_diverse_score(responses_list))

    # Batch process model inputs ONLY if enabled and there are inputs
    model_based_scores = []
    if all_model_inputs and use_model_verifier and vllm_model is not None:
        model_inputs = [model_tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            add_generation_prompt=True,
            tokenize=False
        ) for input_text in all_model_inputs]
        
        outputs = vllm_model.generate(model_inputs, sampling_params)

        for idx, output in enumerate(outputs):
            judgement = output.outputs[0].text.strip()
            model_score = "A" == judgement
            model_based_scores.append(model_score)
    
    # Combine scores
    model_idx = 0
    final_scores = []
    for rule_score in rule_based_scores:
        if rule_score:
            final_scores.append(True)
        else:
            # If rule failed...
            if use_model_verifier:
                # Use the model's judgement
                final_scores.append(model_based_scores[model_idx])
                model_idx += 1
            else:
                # If verifier is disabled, rule failure = final failure
                final_scores.append(False)

    # Calculate metrics
    if final_scores:
        avg_scores = [sum(final_scores[i:i + num_pred]) / num_pred for i in range(0, len(final_scores), num_pred)]
        best = [max(final_scores[i:i + num_pred]) for i in range(0, len(final_scores), num_pred)]
    else:
        avg_scores = [0]
        best = [0]

    solve_none = sum(1 for avg_score in avg_scores if avg_score == 0)
    solve_all = sum(1 for avg_score in avg_scores if avg_score == 1)

    results["mean_score"] = sum(avg_scores) / len(avg_scores) if avg_scores else 0
    results["distinct_4gram"] = sum(diverse) / len(diverse) if diverse else 0
    results["best_score"] = sum(best) / len(best) if best else 0
    results["solve_none"] = solve_none
    results["solve_all"] = solve_all
    results["avg_output_length"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    results["format_error_rollouts"] = without_boxed

    return results

def main():
    parser = argparse.ArgumentParser(description="Grade evaluation results.")
    # New argument to control the verifier
    parser.add_argument(
        "--enable_model_verifier", 
        action="store_true", 
        help="If set, enable the LLM-based verifier. By default, only rule-based grading is used."
    )
    args = parser.parse_args()

    global vllm_model, model_tokenizer, sampling_params
    
    # Only load VLLM if we are enabling the verifier
    if args.enable_model_verifier:
        print("Loading CompassVerifier model...")
        from vllm import LLM, SamplingParams
        model_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        vllm_model = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=8
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048
        )
    else:
        print("Model verifier disabled by default. Running in rule-based only mode.")

    all_results = []
    if not EVAL_DIR.exists():
        print(f"Directory {EVAL_DIR} does not exist.")
        return

    for file_path in EVAL_DIR.glob("*.jsonl"):
        print(f"Processing file: {file_path}")
        # Pass the flag to the grading function
        file_result = grade_file(file_path, use_model_verifier=args.enable_model_verifier)
        if file_result:
            all_results.append(file_result)

    # Save results to JSON
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Grading results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()