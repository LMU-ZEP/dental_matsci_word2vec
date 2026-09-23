#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare unique target-term displacements from 30k and 50k Procrustes alignments."
    )
    p.add_argument(
        "--displacement-30k",
        default="outputs_procrustes_30000/cosine_displacement.csv",
    )
    p.add_argument(
        "--displacement-50k",
        default="outputs_procrustes_50000/cosine_displacement.csv",
    )
    p.add_argument(
        "--output-csv",
        default="outputs_procrustes/procrustes_30k_50k_sensitivity.csv",
    )
    return p.parse_args()


def unique_compared(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"term", "status", "cosine_displacement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return (
        df[df["status"] == "compared"]
        .drop_duplicates(subset=["term"], keep="first")
        .copy()
    )


def main() -> None:
    args = parse_args()
    df30 = unique_compared(args.displacement_30k)
    df50 = unique_compared(args.displacement_50k)

    merged = df30[["term", "cosine_displacement"]].merge(
        df50[["term", "cosine_displacement"]],
        on="term",
        suffixes=("_30k", "_50k"),
        validate="one_to_one",
    )
    merged["abs_diff"] = (
        merged["cosine_displacement_30k"] - merged["cosine_displacement_50k"]
    ).abs()

    pearson_r, pearson_p = pearsonr(
        merged["cosine_displacement_30k"], merged["cosine_displacement_50k"]
    )
    spearman_r, spearman_p = spearmanr(
        merged["cosine_displacement_30k"], merged["cosine_displacement_50k"]
    )

    print("Unique compared terms:", len(merged))
    print("Pearson r:", pearson_r)
    print("Spearman rho:", spearman_r)
    print("Mean absolute difference:", merged["abs_diff"].mean())
    print("Median absolute difference:", merged["abs_diff"].median())
    print("Max absolute difference:", merged["abs_diff"].max())

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)


if __name__ == "__main__":
    main()
