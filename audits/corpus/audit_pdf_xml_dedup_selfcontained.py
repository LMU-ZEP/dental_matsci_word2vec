#!/usr/bin/env python3
"""
audit_pdf_xml_dedup.py

Retrospective audit of cross-source XML/PDF overlap for one period.

It checks:
1) known-year selected PDF records against included XML source records using the historical PDF-selection identity logic:
      - normalized DOI
      - normalized title + publication year
2) unknown-year PDF records against an optional period-specific DMJ folder.
3) whether the XML source manifest itself contains "Dental Materials Journal".

Inputs
------
This version is self-contained and embeds the historical PDF-selection
identity normalization (DOI and title+year).
--pdf-jsonl
    selected_pdf_records_*.jsonl
--xml-included-csv
    retrospective XML audit *.included.csv
--dmj-dir
    optional period-specific DMJ folder used to verify unknown-year provenance

Outputs
-------
<out-prefix>.summary.json
<out-prefix>.cross_source_matches.csv
<out-prefix>.pdf_duplicate_keys.csv
<out-prefix>.unknown_year_not_in_dmj.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any



def normalize_title(title: str | None) -> str | None:
    """Historical PDF-selection normalization."""
    if not title:
        return None
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or None


def normalize_doi(doi: str | None) -> str | None:
    """Historical PDF-selection DOI normalization."""
    if not doi:
        return None
    doi = str(doi).strip().lower()
    doi = doi.removeprefix("doi:")
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.strip(" .;,)")
    return doi or None


def extract_doi_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"(10\.\d{4,9}/[^\s?#]+)", str(url), flags=re.I)
    if not m:
        return None
    doi = re.sub(r"\.pdf$", "", m.group(1), flags=re.I).strip("/")
    return normalize_doi(doi)


def clean(v: Any) -> str:
    return str(v or "").strip()


def safe_year(v: Any) -> int | None:
    if v in (None, "") or isinstance(v, bool):
        return None
    try:
        y = int(v)
        return y if 1800 <= y <= 2100 else None
    except Exception:
        return None


def pdf_year(rec: dict) -> int | None:
    for k in ("year", "publication_year", "publicationYear"):
        y = safe_year(rec.get(k))
        if y is not None:
            return y
    return None


def pdf_filename(rec: dict) -> str:
    for k in (
        "_pdf_path", "pdf_path", "source_path", "file_path",
        "filepath", "path", "filename"
    ):
        v = rec.get(k)
        if v:
            return Path(str(v)).name
    for k in ("paperId", "paper_id"):
        v = rec.get(k)
        if v:
            return f"{v}.pdf"
    return ""


def pdf_doi(rec: dict) -> str | None:
    doi = normalize_doi(rec.get("doi"))
    if doi:
        return doi

    oa = rec.get("openAccessPdf")
    if isinstance(oa, dict):
        url = oa.get("url")
        if url:
            doi = extract_doi_from_url(url)
            if doi:
                return normalize_doi(doi)

    return None


def pdf_keys(rec: dict) -> list[tuple[str, str]]:
    keys = []
    doi = pdf_doi(rec)
    if doi:
        keys.append(("doi", doi))

    year = pdf_year(rec)
    title = clean(rec.get("title"))
    title_key = normalize_title(title)
    if title_key and year:
        keys.append(("title_year", f"{title_key}|{year}"))

    return keys


def xml_keys_from_row(row: dict) -> list[tuple[str, str]]:
    keys = []

    doi = normalize_doi(row.get("doi"))
    if doi:
        keys.append(("doi", doi))

    year = safe_year(row.get("historical_publication_year"))
    title = clean(row.get("article_title"))
    title_key = normalize_title(title)
    if title_key and year:
        keys.append(("title_year", f"{title_key}|{year}"))

    return keys


def norm_journal(s: str) -> str:
    return " ".join(clean(s).casefold().split())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-jsonl", required=True, type=Path)
    p.add_argument("--xml-included-csv", required=True, type=Path)
    p.add_argument("--period", required=True)
    p.add_argument("--dmj-dir", type=Path, default=None)
    p.add_argument("--out-prefix", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # ---------- XML source index ----------
    xml_key_to_rows = defaultdict(list)
    xml_records = 0
    xml_no_identity_keys = 0
    xml_journals = Counter()

    with args.xml_included_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader, start=2):
            xml_records += 1
            journal = norm_journal(row.get("journal_title", ""))
            if journal:
                xml_journals[journal] += 1

            keys = xml_keys_from_row(row)
            if not keys:
                xml_no_identity_keys += 1

            for key in keys:
                xml_key_to_rows[key].append({
                    "row_no": row_no,
                    "filename": row.get("filename", ""),
                    "doi": row.get("doi", ""),
                    "title": row.get("article_title", ""),
                    "year": row.get("historical_publication_year", ""),
                    "journal": row.get("journal_title", ""),
                })

    # ---------- PDF records ----------
    pdf_records = []
    with args.pdf_jsonl.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or "_metadata" in rec:
                continue
            pdf_records.append((line_no, rec))

    known = []
    unknown = []
    pdf_key_to_records = defaultdict(list)

    for line_no, rec in pdf_records:
        y = pdf_year(rec)
        item = {
            "line_no": line_no,
            "record": rec,
            "filename": pdf_filename(rec),
            "year": y,
            "doi": pdf_doi(rec) or "",
            "title": clean(rec.get("title")),
            "journal": clean(
                (rec.get("journal") or {}).get("name")
                if isinstance(rec.get("journal"), dict)
                else rec.get("journal")
            ),
        }

        if y is None:
            unknown.append(item)
        else:
            known.append(item)

        for key in pdf_keys(rec):
            pdf_key_to_records[key].append(item)

    known_no_identity_keys = sum(
        1 for item in known if not pdf_keys(item["record"])
    )

    # ---------- Cross-source overlaps ----------
    matched_pdf_lines = set()
    cross_rows = []

    for item in known:
        for key_type, key_value in pdf_keys(item["record"]):
            xml_matches = xml_key_to_rows.get((key_type, key_value), [])
            if not xml_matches:
                continue
            matched_pdf_lines.add(item["line_no"])
            for x in xml_matches:
                cross_rows.append({
                    "period": args.period,
                    "match_key_type": key_type,
                    "match_key_value": key_value,
                    "pdf_line_no": item["line_no"],
                    "pdf_filename": item["filename"],
                    "pdf_year": item["year"],
                    "pdf_doi": item["doi"],
                    "pdf_title": item["title"],
                    "pdf_journal": item["journal"],
                    "xml_row_no": x["row_no"],
                    "xml_filename": x["filename"],
                    "xml_year": x["year"],
                    "xml_doi": x["doi"],
                    "xml_title": x["title"],
                    "xml_journal": x["journal"],
                })

    # ---------- Duplicate identity keys inside selected PDF ----------
    dup_rows = []
    for (key_type, key_value), items in sorted(pdf_key_to_records.items()):
        unique_lines = sorted({x["line_no"] for x in items})
        if len(unique_lines) <= 1:
            continue
        for item in items:
            dup_rows.append({
                "period": args.period,
                "key_type": key_type,
                "key_value": key_value,
                "pdf_line_no": item["line_no"],
                "pdf_filename": item["filename"],
                "pdf_year": item["year"] if item["year"] is not None else "",
                "pdf_doi": item["doi"],
                "pdf_title": item["title"],
            })

    # ---------- Unknown-year -> DMJ provenance ----------
    dmj_count = None
    unknown_in_dmj = None
    unknown_not_in_dmj = []
    dmj_not_unknown = []

    if args.dmj_dir is not None:
        dmj_names = {
            p.name.casefold(): p.name
            for p in args.dmj_dir.iterdir()
            if p.is_file() and p.suffix.casefold() == ".pdf"
        }
        dmj_count = len(dmj_names)

        unknown_names = {
            item["filename"].casefold(): item
            for item in unknown if item["filename"]
        }

        unknown_not_in_dmj = [
            item for key, item in unknown_names.items()
            if key not in dmj_names
        ]
        unknown_in_dmj = len(unknown_names) - len(unknown_not_in_dmj)
        dmj_not_unknown = [
            original for key, original in dmj_names.items()
            if key not in unknown_names
        ]

    # ---------- Outputs ----------
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cross_csv = Path(str(args.out_prefix) + ".cross_source_matches.csv")
    dup_csv = Path(str(args.out_prefix) + ".pdf_duplicate_keys.csv")
    unknown_csv = Path(str(args.out_prefix) + ".unknown_year_not_in_dmj.csv")
    summary_json = Path(str(args.out_prefix) + ".summary.json")

    cross_fields = [
        "period","match_key_type","match_key_value",
        "pdf_line_no","pdf_filename","pdf_year","pdf_doi","pdf_title","pdf_journal",
        "xml_row_no","xml_filename","xml_year","xml_doi","xml_title","xml_journal"
    ]
    with cross_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cross_fields)
        w.writeheader()
        w.writerows(cross_rows)

    dup_fields = [
        "period","key_type","key_value","pdf_line_no","pdf_filename",
        "pdf_year","pdf_doi","pdf_title"
    ]
    with dup_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dup_fields)
        w.writeheader()
        w.writerows(dup_rows)

    with unknown_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["line_no","filename","doi","title","journal"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in unknown_not_in_dmj:
            w.writerow({k: x[k] for k in fields})

    summary = {
        "period": args.period,
        "pdf_selected_records": len(pdf_records),
        "pdf_known_year_records": len(known),
        "pdf_unknown_year_records": len(unknown),
        "pdf_known_year_records_without_identity_key": known_no_identity_keys,
        "xml_included_source_records": xml_records,
        "xml_records_without_identity_key": xml_no_identity_keys,
        "xml_identity_keys": len(xml_key_to_rows),
        "cross_source": {
            "selected_known_year_pdf_records_matching_xml": len(matched_pdf_lines),
            "match_rows": len(cross_rows),
            "match_by_key_type": dict(Counter(r["match_key_type"] for r in cross_rows)),
        },
        "selected_pdf_internal_identity_duplicates": {
            "duplicate_key_count": sum(
                1 for items in pdf_key_to_records.values()
                if len({x["line_no"] for x in items}) > 1
            ),
            "output_rows": len(dup_rows),
        },
        "dmj_provenance": {
            "dmj_folder": str(args.dmj_dir) if args.dmj_dir else None,
            "dmj_pdf_count": dmj_count,
            "unknown_year_records_with_filename_in_dmj": unknown_in_dmj,
            "unknown_year_records_not_in_dmj": len(unknown_not_in_dmj),
            "dmj_pdfs_not_among_unknown_year_selected_records": len(dmj_not_unknown),
        },
        "xml_dental_materials_journal_records": xml_journals.get(
            "dental materials journal", 0
        ),
        "outputs": {
            "cross_source_matches_csv": str(cross_csv),
            "pdf_duplicate_keys_csv": str(dup_csv),
            "unknown_year_not_in_dmj_csv": str(unknown_csv),
        },
    }

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== PDF/XML DEDUP AUDIT: {args.period} ===")
    print(f"Selected PDF records:                       {len(pdf_records):,}")
    print(f"  known-year:                               {len(known):,}")
    print(f"  unknown-year:                             {len(unknown):,}")
    print(f"Known-year PDF records without identity key:{known_no_identity_keys:>10,}")
    print(f"Included XML source records:                {xml_records:,}")
    print(f"Known-year PDF records matching XML:        {len(matched_pdf_lines):,}")
    print(f"Cross-source match rows:                    {len(cross_rows):,}")
    print(f"PDF internal duplicate identity keys:       {summary['selected_pdf_internal_identity_duplicates']['duplicate_key_count']:,}")
    print(f"XML 'Dental Materials Journal' records:     {summary['xml_dental_materials_journal_records']:,}")

    if args.dmj_dir is not None:
        print(f"DMJ folder PDFs:                            {dmj_count:,}")
        print(f"Unknown-year records found in DMJ folder:   {unknown_in_dmj:,}")
        print(f"Unknown-year records NOT in DMJ folder:     {len(unknown_not_in_dmj):,}")
        print(f"DMJ PDFs not in selected unknown-year set:  {len(dmj_not_unknown):,}")

    print(f"\nSummary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
