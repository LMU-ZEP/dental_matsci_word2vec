#!/usr/bin/env python3
"""
build_retrospective_xml_audit_manifest.py

Reconstruct a SOURCE-RECORD-LEVEL audit manifest for an XML corpus that was
historically built with get_xml_corpus.py.

Purpose
-------
This is a RETROSPECTIVE audit artifact. It must not be described as a manifest
that existed during the original production run.

The script deliberately separates:

1) historical selection logic
   - publication year extracted by the original get_xml_corpus.py
   - unknown year skipped (when --skip-unknown-year is used; default)
   - records with year < --min-year skipped
   - NO upper-year filter is added, because the historical build_xml_corpus()
     did not apply one

2) present-day audit checks
   - optional --audit-max-year flags records outside the declared period
     WITHOUT changing historical inclusion
   - optional metadata columns (DOI, PII, title, journal) are informational only
   - optional exact output verification against the preserved historical
     xml_raw_corpus*.json compares both item count and an order-sensitive
     aggregate content hash

Why this matters
----------------
The production PDF manifests enumerate PDF source files. The XML component was
built separately and then supplied to the combined pipeline as a period-
specific raw JSON corpus. This script reconstructs the XML source-record side
without pretending it was a contemporaneous production manifest.

The original extraction functions are IMPORTED from the preserved historical
get_xml_corpus.py instead of being reimplemented here. This minimizes drift.

Outputs
-------
<out-prefix>.csv
    One row per source XML file.

<out-prefix>.jsonl
    First line: manifest metadata.
    Remaining lines: one row per source XML file.

<out-prefix>.summary.json
    Counts, checks, and verification status.

<out-prefix>.included.csv
    Convenience subset: historically included source XML files only.

Selection-status values
-----------------------
included
excluded_unknown_year
excluded_before_min_year
parse_or_extraction_error

Declared-period audit values
----------------------------
within_declared_period
before_declared_period
after_declared_period
unknown_year

Example
-------
PRE:
python build_retrospective_xml_audit_manifest.py \
  --historical-script scripts/get_xml_corpus.py \
  --input-roots \
    /data/elsevier_acta_biomat_xml_before_2017 \
    /data/elsevier_biomaterials_xml_before_2017 \
    /data/elsevier_dental_materials_xml_before_2017 \
    ... \
  --period-label pre2018 \
  --min-year 1992 \
  --audit-max-year 2017 \
  --expected-included-source-records 50224 \
  --verify-against-raw-corpus xml_coprus/xml_raw_corpus_1992_2017.json \
  --recompute-text-counts \
  --out-prefix outputs_xml_audit/xml_source_audit_manifest_pre2018

POST:
python build_retrospective_xml_audit_manifest.py \
  --historical-script scripts/get_xml_corpus.py \
  --input-roots /data/<post-period XML roots...> \
  --period-label post2018 \
  --min-year 2018 \
  --audit-max-year 2026 \
  --expected-included-source-records 36988 \
  --verify-against-raw-corpus xml_coprus/xml_raw_corpus_after_2018.json \
  --recompute-text-counts \
  --out-prefix outputs_xml_audit/xml_source_audit_manifest_post2018
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_historical_module(path: Path):
    spec = importlib.util.spec_from_file_location("historical_get_xml_corpus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import historical script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = ["extract_publication_year", "extract_elsevier_xml_texts", "local_name"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"Historical script {path} is missing required functions: {missing}"
        )
    return module


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_item_bytes(item: Any) -> bytes:
    """
    Canonical serialization used ONLY for aggregate comparison.
    It is independent of indentation/whitespace in the preserved JSON file.
    """
    return (
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """
    Stream a top-level JSON array using only the Python standard library.
    """
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


def aggregate_json_array(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    for item in iter_json_array(path):
        h.update(stable_json_item_bytes(item))
        n += 1
    return n, h.hexdigest()


def collect_xml_paths(input_roots: list[Path]) -> list[tuple[int, Path, Path]]:
    """
    Preserve historical ordering semantics:
    roots are processed in CLI order; within each root, XML paths are sorted.
    Returns tuples: (root_index, root, xml_path).
    """
    out: list[tuple[int, Path, Path]] = []

    for root_index, root in enumerate(input_roots):
        if root.is_dir():
            paths = sorted(root.glob("**/*.xml"))
        elif root.is_file() and root.suffix.lower() == ".xml":
            paths = [root]
        else:
            raise FileNotFoundError(f"Input root is not a directory/XML file: {root}")

        for path in paths:
            out.append((root_index, root, path))

    return out


def relative_source_path(root: Path, path: Path) -> str:
    if root.is_file():
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def first_text_by_local_names(
    root: ET.Element,
    names: set[str],
) -> tuple[str | None, str | None]:
    names_lower = {x.lower() for x in names}
    for el in root.iter():
        name = local_name(el.tag)
        if name.lower() not in names_lower:
            continue
        text = " ".join(" ".join(el.itertext()).split()).strip()
        if text:
            return text, name
    return None, None


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_doi(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", t, flags=re.I)
    t = re.sub(r"^doi:\s*", "", t, flags=re.I)
    return t.strip().lower() or None


def extract_doi(root: ET.Element) -> tuple[str | None, str | None]:
    for el in root.iter():
        name = local_name(el.tag)
        text = (el.text or "").strip()
        if not text:
            continue

        if name.lower() == "doi":
            doi = normalize_doi(text)
            if doi:
                return doi, name

        if name.lower() == "identifier" and "doi:" in text.lower():
            doi = normalize_doi(text.lower().split("doi:", 1)[1])
            if doi:
                return doi, name

    return None, None


def extract_year_with_provenance(root: ET.Element) -> dict[str, Any]:
    """
    Replicate the historical priority/order exactly, but retain provenance.

    Historical get_xml_corpus.py:
      1. cover-date-year / year-nav / publication-year / copyright-year
      2. coverDate / coverDisplayDate / date-search-begin / date-search-end /
         cover-date-start / cover-date-end
      3. any attribute containing a 19xx/20xx year
    """
    year_candidate_tags = {
        "cover-date-year",
        "year-nav",
        "publication-year",
        "copyright-year",
    }

    date_candidate_tags = {
        "coverDate",
        "coverDisplayDate",
        "date-search-begin",
        "date-search-end",
        "cover-date-start",
        "cover-date-end",
    }

    for el in root.iter():
        name = local_name(el.tag)
        text = (el.text or "").strip()

        if name in year_candidate_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return {
                    "year": int(m.group()),
                    "year_source_kind": "year_tag",
                    "year_source_element": name,
                    "year_source_attribute": "",
                    "year_source_value": text,
                }

    for el in root.iter():
        name = local_name(el.tag)
        text = (el.text or "").strip()

        if name in date_candidate_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return {
                    "year": int(m.group()),
                    "year_source_kind": "date_tag",
                    "year_source_element": name,
                    "year_source_attribute": "",
                    "year_source_value": text,
                }

    for el in root.iter():
        for attr_name, value in el.attrib.items():
            m = re.search(r"\b(19|20)\d{2}\b", str(value))
            if m:
                return {
                    "year": int(m.group()),
                    "year_source_kind": "attribute",
                    "year_source_element": local_name(el.tag),
                    "year_source_attribute": local_name(attr_name),
                    "year_source_value": str(value),
                }

    return {
        "year": None,
        "year_source_kind": "",
        "year_source_element": "",
        "year_source_attribute": "",
        "year_source_value": "",
    }


def declared_period_status(
    year: int | None,
    min_year: int,
    audit_max_year: int | None,
) -> str:
    if year is None:
        return "unknown_year"
    if year < min_year:
        return "before_declared_period"
    if audit_max_year is not None and year > audit_max_year:
        return "after_declared_period"
    return "within_declared_period"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "period",
    "source_order",
    "input_root_index",
    "input_root_label",
    "source_relpath",
    "source_abspath",
    "filename",
    "file_size_bytes",
    "file_sha256",
    "historical_publication_year",
    "year_source_kind",
    "year_source_element",
    "year_source_attribute",
    "year_source_value",
    "historical_selection_status",
    "historically_included",
    "declared_period_status",
    "doi",
    "doi_source_tag",
    "pii",
    "pii_source_tag",
    "article_title",
    "article_title_source_tag",
    "journal_title",
    "journal_title_source_tag",
    "n_extracted_texts",
    "extracted_text_aggregate_sha256",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a retrospective source-record audit manifest for XML corpora "
            "using the preserved historical get_xml_corpus.py logic."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--historical-script",
        type=Path,
        required=True,
        help="Preserved historical get_xml_corpus.py used to build the XML corpus.",
    )
    p.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "XML source directories/files in the SAME ORDER as the historical build. "
            "Within each directory, **/*.xml is sorted exactly as in the historical script."
        ),
    )
    p.add_argument("--period-label", required=True, help="e.g. pre2018 or post2018")
    p.add_argument(
        "--min-year",
        type=int,
        required=True,
        help=(
            "Historical lower-bound selection threshold. For the preserved pre build this is 1992."
        ),
    )
    p.add_argument(
        "--audit-max-year",
        type=int,
        default=None,
        help=(
            "Audit-only declared upper period bound. It NEVER changes historical inclusion."
        ),
    )
    p.add_argument(
        "--skip-unknown-year",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match historical build_xml_corpus(skip_unknown_year=True).",
    )
    p.add_argument(
        "--recompute-text-counts",
        action="store_true",
        help=(
            "Run historical extract_elsevier_xml_texts(..., split='section') "
            "for every historically included XML file and store per-file text counts/hashes."
        ),
    )
    p.add_argument(
        "--verify-against-raw-corpus",
        type=Path,
        default=None,
        help=(
            "Preserved historical xml_raw_corpus*.json. Implies --recompute-text-counts "
            "and compares an order-sensitive aggregate hash of extracted text items."
        ),
    )
    p.add_argument(
        "--expected-included-source-records",
        type=int,
        default=None,
        help="Expected source XML count (e.g. 50224 or current verified post-2018 count 36988).",
    )
    p.add_argument(
        "--include-file-sha256",
        action="store_true",
        help="Hash every source XML file; useful but slower.",
    )
    p.add_argument(
        "--include-absolute-paths",
        action="store_true",
        help=(
            "Include local absolute source paths in output. OFF by default so the "
            "reproduction package does not expose machine-specific paths."
        ),
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output path prefix without extension.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.historical_script.exists():
        raise FileNotFoundError(args.historical_script)

    if args.verify_against_raw_corpus is not None:
        if not args.verify_against_raw_corpus.exists():
            raise FileNotFoundError(args.verify_against_raw_corpus)
        args.recompute_text_counts = True

    hist = load_historical_module(args.historical_script)

    xml_entries = collect_xml_paths(args.input_roots)
    print(f"Collected XML source files: {len(xml_entries):,}")

    rows: list[dict] = []
    counters = Counter()
    recomputed_aggregate = hashlib.sha256()
    recomputed_text_count = 0

    for source_order, (root_index, root_dir, xml_path) in enumerate(xml_entries):
        if source_order == 0 or (source_order + 1) % 1000 == 0:
            print(f"Processed {source_order + 1:,}/{len(xml_entries):,} source XML files")

        row = {name: "" for name in FIELDNAMES}
        row.update({
            "period": args.period_label,
            "source_order": source_order,
            "input_root_index": root_index,
            "input_root_label": root_dir.name,
            "source_relpath": relative_source_path(root_dir, xml_path),
            "source_abspath": str(xml_path.resolve()) if args.include_absolute_paths else "",
            "filename": xml_path.name,
            "file_size_bytes": xml_path.stat().st_size if xml_path.exists() else "",
            "file_sha256": sha256_file(xml_path) if args.include_file_sha256 else "",
            "historically_included": False,
            "n_extracted_texts": "",
            "extracted_text_aggregate_sha256": "",
            "error": "",
        })

        try:
            root = ET.parse(xml_path).getroot()

            # Use historical function as the authoritative year value.
            historical_year = hist.extract_publication_year(root)

            # Independently reconstruct provenance using the same search order.
            year_info = extract_year_with_provenance(root)
            if historical_year != year_info["year"]:
                raise RuntimeError(
                    "Year-provenance reconstruction disagrees with historical function: "
                    f"historical={historical_year}, audit={year_info['year']}"
                )

            row["historical_publication_year"] = (
                historical_year if historical_year is not None else ""
            )
            row["year_source_kind"] = year_info["year_source_kind"]
            row["year_source_element"] = year_info["year_source_element"]
            row["year_source_attribute"] = year_info["year_source_attribute"]
            row["year_source_value"] = year_info["year_source_value"]

            row["declared_period_status"] = declared_period_status(
                historical_year,
                args.min_year,
                args.audit_max_year,
            )

            # EXACT historical inclusion semantics from build_xml_corpus().
            if historical_year is None and args.skip_unknown_year:
                status = "excluded_unknown_year"
                included = False
            elif historical_year is not None and historical_year < args.min_year:
                status = "excluded_before_min_year"
                included = False
            else:
                status = "included"
                included = True

            row["historical_selection_status"] = status
            row["historically_included"] = included

            # Informational metadata only — never used for inclusion.
            doi, doi_tag = extract_doi(root)
            pii, pii_tag = first_text_by_local_names(root, {"pii"})
            title, title_tag = first_text_by_local_names(root, {"title"})
            journal, journal_tag = first_text_by_local_names(
                root,
                {
                    "publicationName",
                    "publication-name",
                    "journal-title",
                    "source-title",
                },
            )

            row["doi"] = doi or ""
            row["doi_source_tag"] = doi_tag or ""
            row["pii"] = pii or ""
            row["pii_source_tag"] = pii_tag or ""
            row["article_title"] = title or ""
            row["article_title_source_tag"] = title_tag or ""
            row["journal_title"] = journal or ""
            row["journal_title_source_tag"] = journal_tag or ""

            if included and args.recompute_text_counts:
                texts = hist.extract_elsevier_xml_texts(xml_path, split="section")
                row["n_extracted_texts"] = len(texts)

                file_text_hash = hashlib.sha256()
                for text_item in texts:
                    b = stable_json_item_bytes(text_item)
                    file_text_hash.update(b)
                    recomputed_aggregate.update(b)
                    recomputed_text_count += 1

                row["extracted_text_aggregate_sha256"] = file_text_hash.hexdigest()

            counters[status] += 1
            counters[f"declared_period.{row['declared_period_status']}"] += 1

        except Exception as exc:
            # Historical build_xml_corpus caught any exception and skipped the file.
            row["historical_selection_status"] = "parse_or_extraction_error"
            row["historically_included"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            counters["parse_or_extraction_error"] += 1

        rows.append(row)

    included_rows = [r for r in rows if r["historically_included"] is True]
    included_count = len(included_rows)

    verification: dict[str, Any] = {
        "recompute_text_counts": bool(args.recompute_text_counts),
        "recomputed_text_count": (
            recomputed_text_count if args.recompute_text_counts else None
        ),
        "recomputed_text_aggregate_sha256": (
            recomputed_aggregate.hexdigest() if args.recompute_text_counts else None
        ),
        "reference_raw_corpus": (
            str(args.verify_against_raw_corpus)
            if args.verify_against_raw_corpus is not None
            else None
        ),
        "reference_raw_item_count": None,
        "reference_raw_aggregate_sha256": None,
        "raw_item_count_match": None,
        "raw_content_order_hash_match": None,
    }

    if args.verify_against_raw_corpus is not None:
        print("Streaming preserved raw XML corpus for exact item/hash comparison...")
        ref_count, ref_hash = aggregate_json_array(args.verify_against_raw_corpus)
        verification.update({
            "reference_raw_item_count": ref_count,
            "reference_raw_aggregate_sha256": ref_hash,
            "raw_item_count_match": (ref_count == recomputed_text_count),
            "raw_content_order_hash_match": (
                ref_hash == recomputed_aggregate.hexdigest()
            ),
        })

    expected_match = None
    if args.expected_included_source_records is not None:
        expected_match = included_count == args.expected_included_source_records

    summary = {
        "manifest_type": "retrospective_xml_source_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": args.period_label,
        "historical_script": str(args.historical_script),
        "historical_script_sha256": sha256_file(args.historical_script),
        "historical_selection_logic": {
            "min_year": args.min_year,
            "skip_unknown_year": bool(args.skip_unknown_year),
            "upper_year_filter_in_historical_logic": False,
            "note": (
                "The preserved historical build_xml_corpus() excluded unknown years "
                "when skip_unknown_year=True and excluded year < min_year. "
                "It did not apply a maximum-year filter."
            ),
        },
        "present_day_audit_only": {
            "audit_max_year": args.audit_max_year,
            "does_not_change_historical_inclusion": True,
            "metadata_fields_do_not_change_historical_inclusion": True,
        },
        "input_roots": [
            {
                "index": i,
                "label": root.name,
                "path": str(root.resolve()) if args.include_absolute_paths else "",
            }
            for i, root in enumerate(args.input_roots)
        ],
        "counts": {
            "source_xml_files_seen": len(rows),
            "historically_included_source_records": included_count,
            "excluded_unknown_year": counters["excluded_unknown_year"],
            "excluded_before_min_year": counters["excluded_before_min_year"],
            "parse_or_extraction_error": counters["parse_or_extraction_error"],
            "declared_period_within": counters["declared_period.within_declared_period"],
            "declared_period_before": counters["declared_period.before_declared_period"],
            "declared_period_after": counters["declared_period.after_declared_period"],
            "declared_period_unknown": counters["declared_period.unknown_year"],
        },
        "expected_included_source_records": args.expected_included_source_records,
        "expected_included_source_records_match": expected_match,
        "verification": verification,
        "interpretation": {
            "retrospective": True,
            "not_a_contemporaneous_production_manifest": True,
            "selection_uses_preserved_historical_logic": True,
            "purpose": (
                "Document XML source-record provenance and allow comparison with "
                "the preserved XML raw corpus without claiming the manifest existed "
                "during the original training run."
            ),
        },
    }

    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(str(prefix) + ".csv")
    included_csv_path = Path(str(prefix) + ".included.csv")
    jsonl_path = Path(str(prefix) + ".jsonl")
    summary_path = Path(str(prefix) + ".summary.json")

    write_csv(csv_path, rows, FIELDNAMES)
    write_csv(included_csv_path, included_rows, FIELDNAMES)

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_metadata": summary}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RETROSPECTIVE XML SOURCE AUDIT ===")
    print(f"period:                         {args.period_label}")
    print(f"source XML files seen:          {len(rows):,}")
    print(f"historically included:          {included_count:,}")
    print(f"excluded unknown year:          {counters['excluded_unknown_year']:,}")
    print(f"excluded before min_year:       {counters['excluded_before_min_year']:,}")
    print(f"parse/extraction errors:        {counters['parse_or_extraction_error']:,}")
    if args.audit_max_year is not None:
        print(
            f"included/seen after audit max:    "
            f"{counters['declared_period.after_declared_period']:,}"
        )

    if args.expected_included_source_records is not None:
        print(
            f"expected included count:         "
            f"{args.expected_included_source_records:,}"
        )
        print(f"expected-count match:            {expected_match}")

    if args.recompute_text_counts:
        print(f"recomputed XML text items:       {recomputed_text_count:,}")

    if args.verify_against_raw_corpus is not None:
        print(f"reference raw JSON items:        {verification['reference_raw_item_count']:,}")
        print(f"item-count match:                {verification['raw_item_count_match']}")
        print(
            f"content/order hash match:        "
            f"{verification['raw_content_order_hash_match']}"
        )

    print(f"\nCSV:      {csv_path}")
    print(f"Included: {included_csv_path}")
    print(f"JSONL:    {jsonl_path}")
    print(f"Summary:  {summary_path}")

    # Non-zero exit only for explicit expected/verification failures.
    failures = []
    if expected_match is False:
        failures.append("included source-record count mismatch")
    if verification["raw_item_count_match"] is False:
        failures.append("raw item-count mismatch")
    if verification["raw_content_order_hash_match"] is False:
        failures.append("raw content/order hash mismatch")

    if failures:
        print("\nAUDIT FAILED: " + "; ".join(failures), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
