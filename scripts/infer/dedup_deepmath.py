import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os

def extract_dapo_question(prompt_list):
    # DAPO prompt format: [{'content': '... Please reason step by step...', 'role': 'user'}]
    if isinstance(prompt_list, (list, np.ndarray)) and len(prompt_list) > 0:
        content = prompt_list[0].get('content', '')
        # Remove common instruction suffixes to get the core question
        suffix = " Please reason step by step, and put your final answer within \\boxed{}."
        if content.endswith(suffix):
            return content[:-len(suffix)].strip()
        return content.strip()
    return ""

def extract_deepmath_question(row):
    # DeepMath format: extra_info['question'] contains the raw question
    extra_info = row.get('extra_info', {})
    if isinstance(extra_info, dict) and 'question' in extra_info:
        return str(extra_info['question']).strip()
    
    # Fallback to prompt content if question not in extra_info
    prompt_list = row.get('prompt', [])
    if isinstance(prompt_list, (list, np.ndarray)) and len(prompt_list) > 0:
        content = prompt_list[0].get('content', '')
        suffix = " Please reason step by step, and put your final answer within \\boxed{{}}."
        if content.endswith(suffix):
            return content[:-len(suffix)].strip()
        return content.strip()
    return ""

def main():
    dapo_path = 'datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet'
    deepmath_path = 'datasets/DeepMath-103K/verl_format/train.parquet'
    output_path = 'datasets/DeepMath-103K/verl_format/train_deduped.parquet'
    
    print(f"Loading DAPO-Math from {dapo_path}...")
    df_dapo = pd.read_parquet(dapo_path)
    dapo_questions = df_dapo['prompt'].apply(extract_dapo_question).tolist()
    print(f"Extracted {len(dapo_questions)} questions from DAPO.")

    print(f"Loading DeepMath from {deepmath_path}...")
    df_deepmath = pd.read_parquet(deepmath_path)
    deepmath_questions = df_deepmath.apply(extract_deepmath_question, axis=1).tolist()
    print(f"Extracted {len(deepmath_questions)} questions from DeepMath.")

    # 1. Exact string deduplication (Phase 1)
    dapo_set = set(dapo_questions)
    is_exact_dup = [q in dapo_set for q in deepmath_questions]
    num_exact_dups = sum(is_exact_dup)
    print(f"Found {num_exact_dups} exact string matches.")

    # 2. Semantic deduplication (Phase 2)
    model_name = 'model/all-mpnet-base-v2'
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Encoding DAPO questions...")
    dapo_embeddings = model.encode(dapo_questions, batch_size=1024, show_progress_bar=True, convert_to_numpy=True)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(dapo_embeddings)
    
    print("Building FAISS index...")
    dimension = dapo_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(dapo_embeddings)
    
    print("Encoding DeepMath questions...")
    deepmath_embeddings = model.encode(deepmath_questions, batch_size=2048, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(deepmath_embeddings)
    
    print("Searching for near duplicates...")
    # Search top-1
    D, I = index.search(deepmath_embeddings, 1)
    
    similarities = D.flatten()
    nearest_indices = I.flatten()
    
    threshold = 0.6
    is_semantic_dup = similarities >= threshold
    
    # Combine exact and semantic dups
    to_remove = np.logical_or(is_exact_dup, is_semantic_dup)
    num_to_remove = np.sum(to_remove)
    
    print(f"\nDeduplication Summary:")
    print(f"Total DeepMath rows: {len(df_deepmath)}")
    print(f"Exact duplicates: {num_exact_dups}")
    print(f"Semantic duplicates (threshold {threshold}): {np.sum(is_semantic_dup) - np.sum(np.logical_and(is_exact_dup, is_semantic_dup))}")
    print(f"Total removed: {num_to_remove}")
    print(f"Remaining: {len(df_deepmath) - num_to_remove}")

    # Output some samples for verification
    print(f"\nSample semantic duplicates (Similarity >= {threshold}):")
    dup_indices = np.where(is_semantic_dup)[0]
    for i in dup_indices[:5]:
        dapo_idx = nearest_indices[i]
        print(f"Similarity: {similarities[i]:.4f}")
        print(f"DeepMath: {deepmath_questions[i][:150]}...")
        print(f"DAPO:     {dapo_questions[dapo_idx][:150]}...")
        print("-" * 50)

    # Save deduped dataset
    df_deduped = df_deepmath[~to_remove].reset_index(drop=True)
    print(f"Saving deduped dataset to {output_path}...")
    df_deduped.to_parquet(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
