import numpy as np
import pandas as pd

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from procrustes_displacement import fit_procrustes, aligned_pre_vector, post_vector, load_wv, read_terms, build_alignment_terms


def evaluate_alignment_same_word_distance(
    pre_wv,
    post_wv,
    train_terms,
    eval_terms,
    R,
    mu_pre,
    mu_post,
):
    rows = []

    for w in eval_terms:
        v_pre_raw = normalize(pre_wv[w].reshape(1, -1).astype(np.float64))[0]
        v_post = normalize(post_wv[w].reshape(1, -1).astype(np.float64))[0]

        v_pre_aligned = (
            (v_pre_raw.reshape(1, -1) - mu_pre) @ R + mu_post
        )[0]

        rows.append({
            "term": w,
            "cosine_distance_before": cosine(v_pre_raw, v_post),
            "cosine_distance_after": cosine(v_pre_aligned, v_post),
            "improvement": cosine(v_pre_raw, v_post) - cosine(v_pre_aligned, v_post),
            "pre_count": pre_wv.get_vecattr(w, "count"),
            "post_count": post_wv.get_vecattr(w, "count"),
        })

    return pd.DataFrame(rows)


def evaluate_self_retrieval(
    pre_wv,
    post_wv,
    eval_terms,
    R,
    mu_pre,
    mu_post,
    candidate_terms=None,
    top_k_values=(1, 5, 10),
):
    if candidate_terms is None:
        candidate_terms = eval_terms

    candidate_terms = list(candidate_terms)

    post_matrix = np.vstack([
        normalize(post_wv[w].reshape(1, -1).astype(np.float64))[0]
        for w in candidate_terms
    ])

    term_to_candidate_idx = {
        w: i for i, w in enumerate(candidate_terms)
    }

    results = []

    for w in eval_terms:
        if w not in term_to_candidate_idx:
            continue

        v_pre_raw = normalize(pre_wv[w].reshape(1, -1).astype(np.float64))
        v_pre_aligned = (v_pre_raw - mu_pre) @ R + mu_post

        sims = cosine_similarity(v_pre_aligned, post_matrix)[0]
        ranked_idx = np.argsort(sims)[::-1]

        correct_idx = term_to_candidate_idx[w]
        rank = int(np.where(ranked_idx == correct_idx)[0][0]) + 1

        row = {
            "term": w,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
        }

        for k in top_k_values:
            row[f"top_{k}"] = rank <= k

        results.append(row)

    return pd.DataFrame(results)


def orthogonality_error(R):
    I = np.eye(R.shape[0])
    return np.linalg.norm(R.T @ R - I, ord="fro")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pre-model", required=True)
    parser.add_argument("--post-model", required=True)
    parser.add_argument("--target-terms", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=30000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    pre_wv = load_wv(args.pre_model)
    post_wv = load_wv(args.post_model)

    print("Reading target terms...")
    target_terms = read_terms(args.target_terms)

    print("Selecting alignment terms...")
    alignment_terms = build_alignment_terms(
        pre_wv=pre_wv,
        post_wv=post_wv,
        target_terms=target_terms,
        min_count=args.min_count,
        top_n=args.top_n,
    )

    print(f"Alignment terms: {len(alignment_terms)}")

    
    # alignment_terms should be your clean shared background vocabulary,
    # excluding target terms.
    train_terms, eval_terms = train_test_split(
        alignment_terms,
        test_size=0.2,
        random_state=42,
    )
    
    # Fit R only on train_terms
    R, mu_pre, mu_post, scale = fit_procrustes(
        pre_wv=pre_wv,
        post_wv=post_wv,
        alignment_terms=train_terms,
    )
    
    # A. Same-word distance
    distance_eval = evaluate_alignment_same_word_distance(
        pre_wv=pre_wv,
        post_wv=post_wv,
        train_terms=train_terms,
        eval_terms=eval_terms,
        R=R,
        mu_pre=mu_pre,
        mu_post=mu_post,
    )
    
    print("Same-word cosine distance diagnostic")
    print("Before mean:", distance_eval["cosine_distance_before"].mean())
    print("After mean:", distance_eval["cosine_distance_after"].mean())
    print("Before median:", distance_eval["cosine_distance_before"].median())
    print("After median:", distance_eval["cosine_distance_after"].median())
    print("Median improvement:", distance_eval["improvement"].median())
    
    # B. Self-retrieval
    # For speed, you can use eval_terms as candidate_terms.
    # A stricter version uses a larger candidate vocabulary.
    retrieval_eval = evaluate_self_retrieval(
        pre_wv=pre_wv,
        post_wv=post_wv,
        eval_terms=eval_terms,
        R=R,
        mu_pre=mu_pre,
        mu_post=mu_post,
        candidate_terms=eval_terms,
    )
    
    print("\nSelf-retrieval diagnostic")
    print("Top-1 accuracy:", retrieval_eval["top_1"].mean())
    print("Top-5 accuracy:", retrieval_eval["top_5"].mean())
    print("Top-10 accuracy:", retrieval_eval["top_10"].mean())
    print("Mean reciprocal rank:", retrieval_eval["reciprocal_rank"].mean())
    print("Median rank:", retrieval_eval["rank"].median())
    
    # C. Orthogonality
    print("\nOrthogonality error:", orthogonality_error(R))
    
    # Save diagnostics
    distance_eval.to_csv(out_dir / "alignment_same_word_distance_eval.csv", index=False)
    retrieval_eval.to_csv(out_dir / "alignment_self_retrieval_eval.csv", index=False)
    
    summary = {
        "n_alignment_train_terms": len(train_terms),
        "n_alignment_eval_terms": len(eval_terms),
        "mean_cosine_before": distance_eval["cosine_distance_before"].mean(),
        "mean_cosine_after": distance_eval["cosine_distance_after"].mean(),
        "median_cosine_before": distance_eval["cosine_distance_before"].median(),
        "median_cosine_after": distance_eval["cosine_distance_after"].median(),
        "median_improvement": distance_eval["improvement"].median(),
        "top1_accuracy": retrieval_eval["top_1"].mean(),
        "top5_accuracy": retrieval_eval["top_5"].mean(),
        "top10_accuracy": retrieval_eval["top_10"].mean(),
        "mean_reciprocal_rank": retrieval_eval["reciprocal_rank"].mean(),
        "median_rank": retrieval_eval["rank"].median(),
        "orthogonality_error": orthogonality_error(R),
    }
    
    pd.DataFrame([summary]).to_csv(out_dir / "alignment_diagnostic_summary.csv",
        index=False,
    )
    
    print("\nAlignment diagnostic summary")
    print(pd.DataFrame([summary]))

if __name__ == "__main__":
    main()
    
    
    
    
