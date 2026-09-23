# procrustes_displacement.py

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize


def load_wv(path):
    """
    Loads either a full Gensim Word2Vec model or KeyedVectors.
    """
    path = str(path)
    try:
        model = Word2Vec.load(path)
        return model.wv
    except Exception:
        return KeyedVectors.load(path)


def read_terms(path):
    """Read target terms from TXT, CSV, or TSV and return unique terms."""
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        terms = df["term"] if "term" in df.columns else df.iloc[:, 0]
    elif path.suffix.lower() in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t")
        terms = df["term"] if "term" in df.columns else df.iloc[:, 0]
    else:
        terms = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term and not term.startswith("#"):
                    terms.append(term)

    return list(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))


def is_clean_token(w):
    """
    Simple filter against obvious PDF/OCR garbage.
    Adjust if needed.
    """
    if len(w) < 3:
        return False
    if re.fullmatch(r"\d+", w):
        return False
    if not re.search(r"[a-zA-Z]", w):
        return False
    return True


def get_count(wv, word):
    try:
        return int(wv.get_vecattr(word, "count"))
    except Exception:
        return 0


def build_alignment_terms(
    pre_wv,
    post_wv,
    target_terms,
    min_count=100,
    top_n=30000,
):
    common_vocab = sorted(set(pre_wv.index_to_key) & set(post_wv.index_to_key))
    target_terms = set(target_terms)

    terms = []

    for w in common_vocab:
        if w in target_terms:
            continue

        if not is_clean_token(w):
            continue

        c_pre = get_count(pre_wv, w)
        c_post = get_count(post_wv, w)

        if c_pre >= min_count and c_post >= min_count:
            terms.append(w)

    terms = sorted(
        terms,
        key=lambda w: min(get_count(pre_wv, w), get_count(post_wv, w)),
        reverse=True,
    )

    return terms[:top_n]


def fit_procrustes(pre_wv, post_wv, alignment_terms):
    X_pre = np.vstack([pre_wv[w] for w in alignment_terms]).astype(np.float64)
    X_post = np.vstack([post_wv[w] for w in alignment_terms]).astype(np.float64)

    X_pre = normalize(X_pre)
    X_post = normalize(X_post)

    mu_pre = X_pre.mean(axis=0, keepdims=True)
    mu_post = X_post.mean(axis=0, keepdims=True)

    X_pre_c = X_pre - mu_pre
    X_post_c = X_post - mu_post

    R, scale = orthogonal_procrustes(X_pre_c, X_post_c)

    return R, mu_pre, mu_post, scale


def aligned_pre_vector(pre_wv, word, R, mu_pre, mu_post):
    v = pre_wv[word].reshape(1, -1).astype(np.float64)
    v = normalize(v)
    v_aligned = (v - mu_pre) @ R + mu_post
    return v_aligned[0]


def post_vector(post_wv, word):
    v = post_wv[word].reshape(1, -1).astype(np.float64)
    v = normalize(v)
    return v[0]


def compute_displacement(pre_wv, post_wv, target_terms, R, mu_pre, mu_post):
    rows = []

    for w in target_terms:
        in_pre = w in pre_wv
        in_post = w in post_wv

        if not in_pre or not in_post:
            rows.append({
                "term": w,
                "status": "missing",
                "in_pre": in_pre,
                "in_post": in_post,
                "cosine_displacement": np.nan,
                "pre_count": get_count(pre_wv, w) if in_pre else 0,
                "post_count": get_count(post_wv, w) if in_post else 0,
            })
            continue

        v_pre_aligned = aligned_pre_vector(pre_wv, w, R, mu_pre, mu_post)
        v_post = post_vector(post_wv, w)

        rows.append({
            "term": w,
            "status": "compared",
            "in_pre": True,
            "in_post": True,
            "cosine_displacement": cosine(v_pre_aligned, v_post),
            "pre_count": get_count(pre_wv, w),
            "post_count": get_count(post_wv, w),
        })

    return pd.DataFrame(rows)


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

    print("Fitting Procrustes alignment...")
    R, mu_pre, mu_post, scale = fit_procrustes(
        pre_wv=pre_wv,
        post_wv=post_wv,
        alignment_terms=alignment_terms,
    )

    print("Computing cosine displacement...")
    displacement_df = compute_displacement(
        pre_wv=pre_wv,
        post_wv=post_wv,
        target_terms=target_terms,
        R=R,
        mu_pre=mu_pre,
        mu_post=mu_post,
    )

    print("Saving outputs...")

    np.savez(
        out_dir / "procrustes_alignment.npz",
        R=R,
        mu_pre=mu_pre,
        mu_post=mu_post,
        scale=scale,
    )

    displacement_df.to_csv(out_dir / "cosine_displacement.csv", index=False)

    with open(out_dir / "alignment_terms.txt", "w", encoding="utf-8") as f:
        for w in alignment_terms:
            f.write(w + "\n")

    metadata = {
        "pre_model": args.pre_model,
        "post_model": args.post_model,
        "target_terms_file": args.target_terms,
        "min_count": args.min_count,
        "top_n": args.top_n,
        "n_alignment_terms": len(alignment_terms),
        "n_target_terms": len(target_terms),
        "n_compared_terms": int((displacement_df["status"] == "compared").sum()),
        "n_missing_terms": int((displacement_df["status"] == "missing").sum()),
        "scale": float(scale),
    }

    with open(out_dir / "procrustes_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(displacement_df)


if __name__ == "__main__":
    main()