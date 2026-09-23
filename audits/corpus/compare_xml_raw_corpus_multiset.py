#!/usr/bin/env python3
"""
Order-independent comparison of a reconstructed XML raw corpus against the
preserved historical raw JSON corpus.

The preserved historical get_xml_corpus.py is imported and used for year
selection and XML text extraction. Comparison is a multiset comparison:
duplicate text items retain their multiplicities.

Outputs:
  <out-prefix>.summary.json
  <out-prefix>.diff_counts.csv

Exit 0: identical multisets
Exit 2: different multisets / item counts
Exit 1: operational error
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def stable_json_item_bytes(item: Any) -> bytes:
    # Must match build_retrospective_xml_audit_manifest.py exactly.
    return (
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def item_sha256(item: Any) -> str:
    return hashlib.sha256(stable_json_item_bytes(item)).hexdigest()


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        eof = False

        while True:
            if pos >= len(buf) and not eof:
                buf = f.read(chunk_size)
                pos = 0
                if not buf:
                    eof = True

            if not started:
                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if pos < len(buf):
                        if buf[pos] != "[":
                            raise ValueError(f"{path}: expected top-level JSON array")
                        pos += 1
                        started = True
                        break
                    if eof:
                        raise ValueError(f"{path}: empty/incomplete JSON")
                    more = f.read(chunk_size)
                    if not more:
                        eof = True
                    buf = buf[pos:] + more
                    pos = 0

            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"{path}: unexpected EOF inside JSON array")
                more = f.read(chunk_size)
                if not more:
                    eof = True
                buf = buf[pos:] + more
                pos = 0

            if buf[pos] == "]":
                return

            while True:
                try:
                    item, end = decoder.raw_decode(buf, pos)
                    pos = end
                    yield item
                    if pos > 4 * chunk_size:
                        buf = buf[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(
                            f"{path}: invalid/incomplete JSON near {exc.pos}: {exc.msg}"
                        ) from exc
                    buf = buf[pos:]
                    pos = 0
                    more = f.read(chunk_size)
                    if not more:
                        eof = True
                    buf += more


def load_historical_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "historical_get_xml_corpus", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("extract_publication_year", "extract_elsevier_xml_texts"):
        if not hasattr(module, name):
            raise AttributeError(f"{script_path} lacks required function {name}")
    return module


def collect_xml_paths(input_roots: list[Path]) -> list[tuple[int, Path, Path]]:
    out = []
    for root_index, root in enumerate(input_roots):
        if root.is_dir():
            paths = sorted(root.glob("**/*.xml"))
        elif root.is_file() and root.suffix.lower() == ".xml":
            paths = [root]
        else:
            raise FileNotFoundError(f"XML input root not found: {root}")
        for path in paths:
            out.append((root_index, root, path))
    return out


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def counter_digest(counter: Counter[str]) -> str:
    """
    Deterministic order-independent digest of item-hash multiplicities.
    """
    h = hashlib.sha256()
    for item_hash in sorted(counter):
        h.update(f"{item_hash}\t{counter[item_hash]}\n".encode("ascii"))
    return h.hexdigest()


def build_reconstructed_counter(
    hist,
    xml_entries,
    min_year: int,
    skip_unknown_year: bool,
):
    counts = Counter()
    stats = Counter()
    total = len(xml_entries)

    for source_order, (_, _, xml_path) in enumerate(xml_entries):
        stats["source_xml_files_seen"] += 1
        if source_order == 0 or (source_order + 1) % 1000 == 0:
            print(f"[reconstructed] {source_order + 1:,}/{total:,} XML files")

        try:
            root = ET.parse(xml_path).getroot()
            year = hist.extract_publication_year(root)

            if year is None and skip_unknown_year:
                stats["excluded_unknown_year"] += 1
                continue
            if year is not None and year < min_year:
                stats["excluded_before_min_year"] += 1
                continue

            texts = hist.extract_elsevier_xml_texts(xml_path, split="section")
            stats["included_source_xml_files"] += 1

            if not texts:
                stats["zero_text_included_xml_files"] += 1

            for text_item in texts:
                counts[item_sha256(text_item)] += 1
                stats["text_items"] += 1

        except Exception as exc:
            stats["parse_or_extraction_error"] += 1
            print(
                f"[WARN] {xml_path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return counts, dict(stats)


def build_reference_counter(reference_path: Path):
    counts = Counter()
    n = 0
    for item in iter_json_array(reference_path):
        counts[item_sha256(item)] += 1
        n += 1
        if n == 1 or n % 100000 == 0:
            print(f"[reference] {n:,} items")
    return counts, n


def make_snippet(text: Any, max_chars: int) -> str:
    s = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def collect_reconstructed_examples(
    hist,
    xml_entries,
    min_year,
    skip_unknown_year,
    wanted_hashes,
    max_chars,
):
    examples = {}
    remaining = set(wanted_hashes)
    if not remaining:
        return examples

    for source_order, (root_index, root_dir, xml_path) in enumerate(xml_entries):
        if not remaining:
            break
        try:
            root = ET.parse(xml_path).getroot()
            year = hist.extract_publication_year(root)
            if year is None and skip_unknown_year:
                continue
            if year is not None and year < min_year:
                continue

            texts = hist.extract_elsevier_xml_texts(xml_path, split="section")
            for text_index, text_item in enumerate(texts):
                h = item_sha256(text_item)
                if h in remaining:
                    examples[h] = {
                        "root_index": root_index,
                        "root_label": root_dir.name,
                        "source_relpath": relpath(root_dir, xml_path),
                        "source_order": source_order,
                        "text_index_in_file": text_index,
                        "text_snippet": make_snippet(text_item, max_chars),
                    }
                    remaining.remove(h)
        except Exception:
            continue
    return examples


def collect_reference_examples(reference_path, wanted_hashes, max_chars):
    examples = {}
    remaining = set(wanted_hashes)
    if not remaining:
        return examples

    for ref_index, item in enumerate(iter_json_array(reference_path)):
        if not remaining:
            break
        h = item_sha256(item)
        if h in remaining:
            examples[h] = {
                "reference_index": ref_index,
                "text_snippet": make_snippet(item, max_chars),
            }
            remaining.remove(h)
    return examples


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare current XML reconstruction with preserved raw corpus "
            "as text-item multisets, ignoring order."
        )
    )
    p.add_argument("--historical-script", required=True, type=Path)
    p.add_argument("--input-roots", required=True, nargs="+", type=Path)
    p.add_argument("--min-year", required=True, type=int)
    p.add_argument(
        "--skip-unknown-year",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--reference-raw-corpus", required=True, type=Path)
    p.add_argument("--out-prefix", required=True, type=Path)
    p.add_argument(
        "--max-diff-rows",
        type=int,
        default=5000,
        help="0 = write all differing hashes",
    )
    p.add_argument("--max-example-chars", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.historical_script.exists():
        print(f"ERROR: missing {args.historical_script}", file=sys.stderr)
        return 1
    if not args.reference_raw_corpus.exists():
        print(f"ERROR: missing {args.reference_raw_corpus}", file=sys.stderr)
        return 1
    for root in args.input_roots:
        if not root.exists():
            print(f"ERROR: missing input root {root}", file=sys.stderr)
            return 1

    hist = load_historical_module(args.historical_script)
    xml_entries = collect_xml_paths(args.input_roots)
    print(f"Collected current XML source files: {len(xml_entries):,}")

    reconstructed, source_stats = build_reconstructed_counter(
        hist,
        xml_entries,
        args.min_year,
        args.skip_unknown_year,
    )
    reference, _ = build_reference_counter(args.reference_raw_corpus)

    matched = reconstructed & reference
    only_reconstructed = reconstructed - reference
    only_reference = reference - reconstructed

    reconstructed_n = sum(reconstructed.values())
    reference_n = sum(reference.values())
    matched_n = sum(matched.values())
    only_reconstructed_n = sum(only_reconstructed.values())
    only_reference_n = sum(only_reference.values())

    reconstructed_digest = counter_digest(reconstructed)
    reference_digest = counter_digest(reference)
    multiset_equal = reconstructed == reference

    diff_rows = []
    for h in set(only_reconstructed) | set(only_reference):
        rc = reconstructed.get(h, 0)
        fc = reference.get(h, 0)
        diff_rows.append(
            {
                "item_sha256": h,
                "reconstructed_count": rc,
                "reference_count": fc,
                "delta_reconstructed_minus_reference": rc - fc,
            }
        )

    diff_rows.sort(
        key=lambda r: (
            -abs(r["delta_reconstructed_minus_reference"]),
            r["item_sha256"],
        )
    )

    if args.max_diff_rows != 0:
        rows_to_write = diff_rows[: args.max_diff_rows]
    else:
        rows_to_write = diff_rows

    selected_hashes = {r["item_sha256"] for r in rows_to_write}

    re_examples = collect_reconstructed_examples(
        hist,
        xml_entries,
        args.min_year,
        args.skip_unknown_year,
        selected_hashes & set(only_reconstructed),
        args.max_example_chars,
    )
    ref_examples = collect_reference_examples(
        args.reference_raw_corpus,
        selected_hashes & set(only_reference),
        args.max_example_chars,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(str(args.out_prefix) + ".summary.json")
    diff_path = Path(str(args.out_prefix) + ".diff_counts.csv")

    fields = [
        "item_sha256",
        "reconstructed_count",
        "reference_count",
        "delta_reconstructed_minus_reference",
        "reconstructed_root_index",
        "reconstructed_root_label",
        "reconstructed_source_relpath",
        "reconstructed_source_order",
        "reconstructed_text_index_in_file",
        "reconstructed_text_snippet",
        "reference_index",
        "reference_text_snippet",
    ]

    with diff_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_to_write:
            h = row["item_sha256"]
            rex = re_examples.get(h, {})
            fex = ref_examples.get(h, {})
            w.writerow(
                {
                    **row,
                    "reconstructed_root_index": rex.get("root_index", ""),
                    "reconstructed_root_label": rex.get("root_label", ""),
                    "reconstructed_source_relpath": rex.get("source_relpath", ""),
                    "reconstructed_source_order": rex.get("source_order", ""),
                    "reconstructed_text_index_in_file": rex.get(
                        "text_index_in_file", ""
                    ),
                    "reconstructed_text_snippet": rex.get("text_snippet", ""),
                    "reference_index": fex.get("reference_index", ""),
                    "reference_text_snippet": fex.get("text_snippet", ""),
                }
            )

    summary = {
        "comparison_type": "order_independent_text_item_multiset",
        "historical_script": str(args.historical_script),
        "reference_raw_corpus": str(args.reference_raw_corpus),
        "min_year": args.min_year,
        "skip_unknown_year": bool(args.skip_unknown_year),
        "input_roots": [
            {"index": i, "label": p.name}
            for i, p in enumerate(args.input_roots)
        ],
        "reconstructed_source_stats": source_stats,
        "comparison": {
            "reconstructed_item_count": reconstructed_n,
            "reference_item_count": reference_n,
            "item_count_match": reconstructed_n == reference_n,
            "reconstructed_unique_item_hashes": len(reconstructed),
            "reference_unique_item_hashes": len(reference),
            "matched_item_count_with_multiplicity": matched_n,
            "only_reconstructed_item_count_with_multiplicity": only_reconstructed_n,
            "only_reference_item_count_with_multiplicity": only_reference_n,
            "only_reconstructed_unique_hashes": len(only_reconstructed),
            "only_reference_unique_hashes": len(only_reference),
            "reconstructed_multiset_sha256": reconstructed_digest,
            "reference_multiset_sha256": reference_digest,
            "multiset_hash_match": reconstructed_digest == reference_digest,
            "multiset_equal": multiset_equal,
        },
        "difference_output": {
            "total_differing_unique_hashes": len(diff_rows),
            "diff_rows_written": len(rows_to_write),
            "diff_rows_truncated": len(rows_to_write) < len(diff_rows),
            "diff_csv": str(diff_path),
        },
        "interpretation": (
            "The preserved and reconstructed corpora contain exactly the same "
            "text items with the same multiplicities. Any order-sensitive hash "
            "difference is therefore attributable to ordering."
            if multiset_equal
            else
            "The preserved and reconstructed corpora differ as text-item "
            "multisets. Inspect the diff CSV for missing/additional items or "
            "multiplicity differences."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ORDER-INDEPENDENT XML CORPUS COMPARISON ===")
    print(f"source XML files seen:             {source_stats.get('source_xml_files_seen', 0):,}")
    print(f"included source XML files:         {source_stats.get('included_source_xml_files', 0):,}")
    print(f"zero-text included XML files:      {source_stats.get('zero_text_included_xml_files', 0):,}")
    print(f"reconstructed text items:          {reconstructed_n:,}")
    print(f"reference text items:              {reference_n:,}")
    print(f"item-count match:                  {reconstructed_n == reference_n}")
    print(f"matched items (with multiplicity): {matched_n:,}")
    print(f"only reconstructed items:          {only_reconstructed_n:,}")
    print(f"only reference items:              {only_reference_n:,}")
    print(f"multiset hash match:               {reconstructed_digest == reference_digest}")
    print(f"multiset equal:                    {multiset_equal}")
    print(f"\nSummary: {summary_path}")
    print(f"Diff CSV: {diff_path}")

    if multiset_equal:
        print("\nRESULT: SAME CONTENT/MULTIPLICITIES; ORDER IGNORED.")
        return 0

    print("\nRESULT: CONTENT/MULTIPLICITY DIFFERENCE EXISTS.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
