#!/usr/bin/env python3
"""Create canonical Procrustes displacement summaries for the manuscript.

This version keeps the validated global/reliability logic of the historical
``calculate_procrustes_summary.py`` but removes hard-coded material-system
term lists. Material-system membership is read from ``target_terms.tsv``,
which is therefore the single source of truth for group-level summaries.

Default inputs
--------------
- outputs_procrustes_30000/cosine_displacement.csv
- target_terms.tsv

Default outputs
---------------
- outputs_procrustes_methods/cosine_displacement_global_summary.csv
- outputs_procrustes_methods/cosine_displacement_group_summary.csv
- outputs_procrustes_methods/cosine_displacement_unique_terms.csv
- outputs_procrustes_methods/cosine_displacement_reliable_unique_terms.csv
- outputs_procrustes_methods/cosine_displacement_with_material_groups.csv

The reliability criterion is min(pre_count, post_count) >= 1000 by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DISPLACEMENT_CSV = "outputs_procrustes_30000/cosine_displacement.csv"
DEFAULT_TARGET_TERMS_TSV = "target_terms.tsv"
DEFAULT_OUT_DIR = "outputs_procrustes_methods"
DEFAULT_RELIABLE_MIN_COUNT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Procrustes cosine displacement globally and by material "
            "system using target_terms.tsv as the canonical group definition."
        )
    )
    parser.add_argument(
        "--displacement-csv",
        default=DEFAULT_DISPLACEMENT_CSV,
        help=f"Input displacement CSV (default: {DEFAULT_DISPLACEMENT_CSV})",
    )
    parser.add_argument(
        "--target-terms",
        "--target-terms-tsv",
        dest="target_terms_tsv",
        default=DEFAULT_TARGET_TERMS_TSV,
        help=f"TSV with columns term, material_system (default: {DEFAULT_TARGET_TERMS_TSV})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--reliable-min-count",
        type=int,
        default=DEFAULT_RELIABLE_MIN_COUNT,
        help=(
            "Reliability threshold applied to min(pre_count, post_count) "
            f"(default: {DEFAULT_RELIABLE_MIN_COUNT})"
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s): {', '.join(missing)}"
        )


def check_duplicate_displacements_are_consistent(df: pd.DataFrame) -> None:
    """Fail if duplicate rows for one term disagree on scientific values.

    Historical displacement files may contain repeated target rows. Those
    duplicate rows must carry identical displacement
    and count values before they are collapsed to one term for global summaries
    and then re-expanded through target_terms.tsv for material-system summaries.
    """

    duplicated = df[df.duplicated(subset=["term"], keep=False)].copy()
    if duplicated.empty:
        return

    value_columns = [
        "status",
        "in_pre",
        "in_post",
        "cosine_displacement",
        "pre_count",
        "post_count",
    ]

    inconsistent_terms: list[str] = []
    for term, group in duplicated.groupby("term", sort=True):
        # Treat NaN values in the same position as equal by comparing normalized
        # string representations after replacing NaN with a sentinel.
        normalized = group[value_columns].copy()
        normalized = normalized.where(normalized.notna(), "<NA>")
        if len(normalized.drop_duplicates()) > 1:
            inconsistent_terms.append(str(term))

    if inconsistent_terms:
        raise ValueError(
            "Duplicate displacement rows disagree for term(s): "
            + ", ".join(inconsistent_terms)
        )


def summarize(data: pd.DataFrame, label: str, group_col: str | None = None) -> pd.DataFrame:
    data = data.copy()

    if group_col is None:
        data["_group"] = "all_terms"
        group_col = "_group"

    summary = (
        data.groupby(group_col, dropna=False)
        .agg(
            n_terms=("term", "count"),
            mean_displacement=("cosine_displacement", "mean"),
            median_displacement=("cosine_displacement", "median"),
            std_displacement=("cosine_displacement", "std"),
            min_displacement=("cosine_displacement", "min"),
            max_displacement=("cosine_displacement", "max"),
            mean_min_count=("min_count", "mean"),
            median_min_count=("min_count", "median"),
        )
        .reset_index()
    )
    summary.insert(0, "summary_variant", label)
    return summary


def main() -> None:
    args = parse_args()

    displacement_path = Path(args.displacement_csv)
    target_terms_path = Path(args.target_terms_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and validate displacement data
    # ------------------------------------------------------------------
    df = pd.read_csv(displacement_path)
    require_columns(
        df,
        {
            "term",
            "status",
            "in_pre",
            "in_post",
            "cosine_displacement",
            "pre_count",
            "post_count",
        },
        str(displacement_path),
    )

    df["term"] = df["term"].astype(str).str.strip()
    if (df["term"] == "").any():
        raise ValueError(f"{displacement_path} contains blank term values")

    df["pre_count"] = pd.to_numeric(df["pre_count"], errors="raise")
    df["post_count"] = pd.to_numeric(df["post_count"], errors="raise")
    df["cosine_displacement"] = pd.to_numeric(
        df["cosine_displacement"], errors="coerce"
    )

    df["min_count"] = df[["pre_count", "post_count"]].min(axis=1)
    df["max_count"] = df[["pre_count", "post_count"]].max(axis=1)

    check_duplicate_displacements_are_consistent(df)

    # Only rows with a valid paired displacement enter quantitative summaries.
    df_compared = df[df["status"] == "compared"].copy()

    # Preserve the historical global-summary behavior: the first variant counts
    # all compared rows, including repeated terms; the manuscript-level global
    # analysis uses the unique-term variant below.
    summary_all_rows = summarize(
        df_compared,
        label="all_rows_including_duplicates",
        group_col=None,
    )

    # One scientific displacement value per unique target term.
    df_unique = (
        df_compared.sort_values("term")
        .drop_duplicates(subset=["term"], keep="first")
        .reset_index(drop=True)
    )

    summary_unique = summarize(
        df_unique,
        label="unique_terms",
        group_col=None,
    )

    df_reliable_unique = df_unique[
        df_unique["min_count"] >= args.reliable_min_count
    ].copy()

    summary_reliable_unique = summarize(
        df_reliable_unique,
        label=f"unique_terms_min_count_ge_{args.reliable_min_count}",
        group_col=None,
    )

    global_summary = pd.concat(
        [summary_all_rows, summary_unique, summary_reliable_unique],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 2. Load canonical material-system membership from target_terms.tsv
    # ------------------------------------------------------------------
    groups = pd.read_csv(target_terms_path, sep="\t")
    require_columns(groups, {"term", "material_system"}, str(target_terms_path))

    groups = groups[["term", "material_system"]].copy()
    groups["term"] = groups["term"].astype(str).str.strip()
    groups["material_system"] = groups["material_system"].astype(str).str.strip()

    if (groups["term"] == "").any() or (groups["material_system"] == "").any():
        raise ValueError(f"{target_terms_path} contains blank term/material_system values")

    # Duplicate membership rows inside the same material system would inflate n.
    # Exact duplicate memberships are therefore collapsed; cross-system reuse of
    # the same term is retained intentionally.
    groups = (
        groups.drop_duplicates(subset=["material_system", "term"], keep="first")
        .sort_values(["material_system", "term"])
        .reset_index(drop=True)
    )

    # Use one displacement row per term before joining to material groups.  This
    # prevents a term repeated in the original displacement CSV from being
    # multiplied again during the merge.
    df_unique_all_status = (
        df.sort_values("term")
        .drop_duplicates(subset=["term"], keep="first")
        .reset_index(drop=True)
    )

    df_grouped = groups.merge(
        df_unique_all_status,
        on="term",
        how="left",
        validate="many_to_one",
    )

    # A canonical target-term definition should have a corresponding row in the
    # displacement output, even if that term has status=missing in one model.
    absent_from_displacement = df_grouped[df_grouped["status"].isna()]["term"].unique()
    if len(absent_from_displacement) > 0:
        raise ValueError(
            "target_terms.tsv contains term(s) absent from the displacement CSV: "
            + ", ".join(sorted(map(str, absent_from_displacement)))
        )

    df_grouped_compared = df_grouped[df_grouped["status"] == "compared"].copy()

    # Because target_terms.tsv is now the single source of truth, a term may
    # correctly appear in more than one material system (e.g. translucency or
    # fracture_toughness), but only once within each system.
    group_summary_all = summarize(
        df_grouped_compared,
        label="all_group_terms",
        group_col="material_system",
    )

    df_grouped_unique = (
        df_grouped_compared.sort_values(["material_system", "term"])
        .drop_duplicates(subset=["material_system", "term"], keep="first")
        .reset_index(drop=True)
    )

    group_summary_unique = summarize(
        df_grouped_unique,
        label="unique_terms_within_group",
        group_col="material_system",
    )

    df_grouped_reliable_unique = df_grouped_unique[
        df_grouped_unique["min_count"] >= args.reliable_min_count
    ].copy()

    group_summary_reliable = summarize(
        df_grouped_reliable_unique,
        label=(
            "unique_terms_within_group_min_count_ge_"
            f"{args.reliable_min_count}"
        ),
        group_col="material_system",
    )

    group_summary = pd.concat(
        [group_summary_all, group_summary_unique, group_summary_reliable],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    output_paths = {
        "global": out_dir / "cosine_displacement_global_summary.csv",
        "group": out_dir / "cosine_displacement_group_summary.csv",
        "unique": out_dir / "cosine_displacement_unique_terms.csv",
        "reliable_unique": out_dir / "cosine_displacement_reliable_unique_terms.csv",
        "grouped": out_dir / "cosine_displacement_with_material_groups.csv",
    }

    global_summary.to_csv(output_paths["global"], index=False)
    group_summary.to_csv(output_paths["group"], index=False)
    df_unique.to_csv(output_paths["unique"], index=False)
    df_reliable_unique.to_csv(output_paths["reliable_unique"], index=False)
    df_grouped.to_csv(output_paths["grouped"], index=False)

    print("\nGLOBAL SUMMARY")
    print(global_summary.to_string(index=False))

    print("\nGROUP SUMMARY")
    print(group_summary.to_string(index=False))

    print("\nCOUNTS")
    print(f"Compared rows (including repeated terms): {len(df_compared)}")
    print(f"Unique compared terms: {len(df_unique)}")
    print(
        f"Reliable unique terms (min_count >= {args.reliable_min_count}): "
        f"{len(df_reliable_unique)}"
    )

    print("\nSaved:")
    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
