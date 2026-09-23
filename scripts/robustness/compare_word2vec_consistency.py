#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "term"


def read_anchor_terms(path: str | Path | None, extra_terms: list[str] | None) -> list[str]:
    terms: list[str] = []
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    if extra_terms:
        terms.extend(extra_terms)
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    if not out:
        raise ValueError("No anchor terms provided. Use --anchor-file or --anchor-terms.")
    return out


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_count(model: Word2Vec, term: str) -> int | None:
    if term not in model.wv:
        return None
    return int(model.wv.get_vecattr(term, "count"))


def top_neighbors(model: Word2Vec, term: str, topn: int) -> list[tuple[str, float]]:
    if term not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(term, topn=topn)]


def jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def mean_abs_rank_displacement(full_neighbors: list[str], sample_neighbors: list[str]) -> float | None:
    full_rank = {w: i + 1 for i, w in enumerate(full_neighbors)}
    sample_rank = {w: i + 1 for i, w in enumerate(sample_neighbors)}
    common = set(full_rank) & set(sample_rank)
    if not common:
        return None
    return float(np.mean([abs(full_rank[w] - sample_rank[w]) for w in common]))


def pearson_corr(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 2:
        return None
    return pearson_corr(rank_values(x), rank_values(y))


def pairwise_similarity_rows(full_model: Word2Vec, sample_model: Word2Vec, terms: list[str], label: str):
    terms = sorted({t for t in terms if t in full_model.wv and t in sample_model.wv})
    rows = []
    full_sims = []
    sample_sims = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            t1, t2 = terms[i], terms[j]
            sf = float(full_model.wv.similarity(t1, t2))
            ss = float(sample_model.wv.similarity(t1, t2))
            rows.append({
                "label": label,
                "term_1": t1,
                "term_2": t2,
                "similarity_full": sf,
                "similarity_sample": ss,
                "abs_difference": abs(sf - ss),
            })
            full_sims.append(sf)
            sample_sims.append(ss)
    summary = {
        "label": label,
        "n_terms_in_pairwise_check": len(terms),
        "n_pairs": len(rows),
        "pearson_pairwise_similarity": pearson_corr(full_sims, sample_sims),
        "spearman_pairwise_similarity": spearman_corr(full_sims, sample_sims),
        "mean_abs_pairwise_similarity_difference": float(np.mean([r["abs_difference"] for r in rows])) if rows else None,
        "median_abs_pairwise_similarity_difference": float(np.median([r["abs_difference"] for r in rows])) if rows else None,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare full Word2Vec models with 90% consistency-check models.")
    p.add_argument("--full-models", nargs="+", required=True)
    p.add_argument("--sample-models", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--anchor-file", default=None)
    p.add_argument("--anchor-terms", nargs="*", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    p.add_argument("--expand-neighbors", type=int, default=20)
    p.add_argument("--max-pairwise-terms", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    full_paths = [Path(p) for p in args.full_models]
    sample_paths = [Path(p) for p in args.sample_models]
    labels = args.labels
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (len(full_paths) == len(sample_paths) == len(labels)):
        raise ValueError("--full-models, --sample-models, and --labels must have the same length.")

    anchors = read_anchor_terms(args.anchor_file, args.anchor_terms)
    max_k = max(args.top_k)
    print(f"Anchor terms: {len(anchors)}")

    all_neighbor_rows = []
    all_pairwise_rows = []
    summaries = []

    for full_path, sample_path, label in zip(full_paths, sample_paths, labels):
        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print("full model:", full_path)
        print("sample model:", sample_path)
        full_model = Word2Vec.load(str(full_path))
        sample_model = Word2Vec.load(str(sample_path))

        neighbor_details_dir = out_dir / "neighbor_details" / label
        neighbor_details_dir.mkdir(parents=True, exist_ok=True)
        expanded_terms = set(anchors)
        label_rows = []

        for anchor in anchors:
            in_full = anchor in full_model.wv
            in_sample = anchor in sample_model.wv
            row = {
                "label": label,
                "anchor": anchor,
                "in_full_model": in_full,
                "in_sample_model": in_sample,
                "count_full": model_count(full_model, anchor),
                "count_sample": model_count(sample_model, anchor),
            }
            if not in_full or not in_sample:
                row.update({"status": "missing_anchor", "top_full": "", "top_sample": ""})
                for k in args.top_k:
                    row[f"overlap_at_{k}"] = None
                    row[f"jaccard_at_{k}"] = None
                all_neighbor_rows.append(row)
                label_rows.append(row)
                continue

            full_nn = top_neighbors(full_model, anchor, max_k)
            sample_nn = top_neighbors(sample_model, anchor, max_k)
            full_words = [w for w, _ in full_nn]
            sample_words = [w for w, _ in sample_nn]
            expanded_terms.update(full_words[: args.expand_neighbors])
            expanded_terms.update(sample_words[: args.expand_neighbors])

            row.update({
                "status": "ok",
                "top_full": "; ".join(full_words[:10]),
                "top_sample": "; ".join(sample_words[:10]),
                "mean_abs_rank_displacement_common_top_max_k": mean_abs_rank_displacement(full_words, sample_words),
            })
            for k in args.top_k:
                a = set(full_words[:k])
                b = set(sample_words[:k])
                row[f"overlap_at_{k}"] = len(a & b)
                row[f"jaccard_at_{k}"] = jaccard(a, b)
            all_neighbor_rows.append(row)
            label_rows.append(row)

            full_rank = {w: i + 1 for i, w in enumerate(full_words)}
            sample_rank = {w: i + 1 for i, w in enumerate(sample_words)}
            full_sim = {w: s for w, s in full_nn}
            sample_sim = {w: s for w, s in sample_nn}
            detail_rows = []
            for w in sorted(set(full_words) | set(sample_words)):
                detail_rows.append({
                    "label": label,
                    "anchor": anchor,
                    "neighbor": w,
                    "rank_full": full_rank.get(w),
                    "rank_sample": sample_rank.get(w),
                    "similarity_full": full_sim.get(w),
                    "similarity_sample": sample_sim.get(w),
                    "in_both_top_max_k": w in full_rank and w in sample_rank,
                })
            write_csv(detail_rows, neighbor_details_dir / f"{safe_filename(anchor)}.neighbors_full_vs_sample.csv")

        expanded_terms = [t for t in expanded_terms if t in full_model.wv and t in sample_model.wv]
        if len(expanded_terms) > args.max_pairwise_terms:
            anchor_set = set(anchors)
            non_anchor_terms = [t for t in expanded_terms if t not in anchor_set]
            non_anchor_terms = sorted(non_anchor_terms, key=lambda t: full_model.wv.get_vecattr(t, "count"), reverse=True)
            keep_n = max(0, args.max_pairwise_terms - len(anchor_set))
            expanded_terms = list(anchor_set) + non_anchor_terms[:keep_n]

        pair_rows, pair_summary = pairwise_similarity_rows(full_model, sample_model, expanded_terms, label)
        all_pairwise_rows.extend(pair_rows)
        ok_rows = [r for r in label_rows if r.get("status") == "ok"]
        summary = {
            "label": label,
            "full_model": full_path.as_posix(),
            "sample_model": sample_path.as_posix(),
            "n_anchors_total": len(anchors),
            "n_anchors_present_in_both": len(ok_rows),
            "top_k": args.top_k,
            "mean_jaccard": {str(k): float(np.mean([r[f"jaccard_at_{k}"] for r in ok_rows])) if ok_rows else None for k in args.top_k},
            "median_jaccard": {str(k): float(np.median([r[f"jaccard_at_{k}"] for r in ok_rows])) if ok_rows else None for k in args.top_k},
            "mean_overlap": {str(k): float(np.mean([r[f"overlap_at_{k}"] for r in ok_rows])) if ok_rows else None for k in args.top_k},
            "pairwise_similarity_consistency": pair_summary,
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_csv(all_neighbor_rows, out_dir / "nearest_neighbor_consistency.csv")
    write_csv(all_pairwise_rows, out_dir / "pairwise_similarity_consistency.csv")
    (out_dir / "consistency_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved:")
    print(" ", out_dir / "nearest_neighbor_consistency.csv")
    print(" ", out_dir / "pairwise_similarity_consistency.csv")
    print(" ", out_dir / "consistency_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
