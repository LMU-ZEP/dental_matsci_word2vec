#!/usr/bin/env python3
"""
audit_journal_statistics_canonical.py

Canonical journal-level audit for the final pre/post XML + PDF source-record sets.

This version combines the stronger parts of the earlier JSONL-based journal audit
with the provenance established in the later corpus audit:

  * XML input is the retrospective INCLUDED source-record manifest CSV produced by
    build_retrospective_xml_audit_manifest.py, rather than a separate XML JSONL.
  * PDF input is the selected PDF-record JSONL actually used for the corpus.
  * Missing PDF journal metadata can be resolved as "Dental Materials Journal"
    only when the selected PDF filename is present in the corresponding audited,
    period-specific DMJ source folder.
  * The expected post-2018 XML source-record count is 36,988 (not 36,998).
  * Alias mapping, explicit exclusions, non-journal review flags, optional XML-file
    fallback, record-level audit output, and Table S2 helpers are retained.

Important
---------
Each included XML-manifest row and each selected PDF JSONL row is treated as one
SOURCE RECORD. This script does NOT re-select or bibliographically deduplicate the
historical corpus. Counts therefore characterize source-record composition and must
not be described as counts of deduplicated unique publications.

Base journal normalization is deliberately conservative:
  Unicode NFKC -> casefold -> '&' to 'and' -> punctuation/symbols to spaces ->
  whitespace collapse.
Known journal-title variants must be supplied explicitly through --aliases.
Potential non-journal labels are flagged for review but are not automatically removed;
use --exclude-titles after review.

Inputs
------
Required:
  --pre-xml-manifest   pre-2018 *.included.csv
  --post-xml-manifest  post-2018 *.included.csv
  --pre-pdf-jsonl      selected_pdf_records_1992_2017.jsonl
  --post-pdf-jsonl     selected_pdf_records_after_2018.jsonl
  --pre-dmj-dir        period-specific DMJ PDF directory used for pre-2018 extras
  --post-dmj-dir       period-specific DMJ PDF directory used for post-2018 extras

Optional:
  --aliases            TSV: alias<TAB>canonical
  --exclude-titles     text file: one excluded source/journal title per line
  --path-journal-map   TSV: path_substring<TAB>journal_title (last-resort mapping)
  --xml-fallback-to-files
                       parse original XML files when manifest journal_title is blank
                       and source_abspath is available
  --panel-b-journals   text file: one selected dental/restorative journal per line

Outputs
-------
  journal_counts.csv
  table_s2_panel_a.csv
  table_s2_panel_b.csv                 (if --panel-b-journals is supplied)
  journal_title_mapping.csv
  journal_record_audit.csv
  missing_journal_records.csv
  journal_review_candidates.csv
  xml_fallback_tag_usage.csv
  xml_root_counts.csv
  dmj_provenance_audit.csv
  summary.json
  journal_aliases_template.tsv
  journal_exclusions_template.txt
  path_journal_map_template.tsv
  panel_b_journals_template.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def clean_metadata_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().casefold()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        m = re.search(r"\b(19|20)\d{2}\b", str(value))
        return int(m.group()) if m else None


def nested_get(obj: dict, *path: str) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def value_to_title(value: Any) -> str | None:
    """Convert common JSON representations of a journal/container title to one string."""
    if value is None:
        return None
    if isinstance(value, str):
        s = clean_metadata_text(value)
        return s or None
    if isinstance(value, dict):
        for k in ("name", "title", "display_name", "publicationName", "publication_name"):
            if k in value:
                s = value_to_title(value[k])
                if s:
                    return s
        return None
    if isinstance(value, (list, tuple)):
        vals = []
        for x in value:
            s = value_to_title(x)
            if s:
                vals.append(s)
        vals = list(dict.fromkeys(vals))
        return vals[0] if len(vals) == 1 else None
    s = clean_metadata_text(value)
    return s or None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------
# PDF JSONL field extraction
# ---------------------------------------------------------------------

JOURNAL_CANDIDATE_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("journal.name", ("journal", "name")),
    ("journal.title", ("journal", "title")),
    ("journal", ("journal",)),
    ("publicationName", ("publicationName",)),
    ("publication_name", ("publication_name",)),
    ("publication-title", ("publication-title",)),
    ("publication_title", ("publication_title",)),
    ("journalTitle", ("journalTitle",)),
    ("journal_title", ("journal_title",)),
    ("sourceTitle", ("sourceTitle",)),
    ("source_title", ("source_title",)),
    ("source.name", ("source", "name")),
    ("source.title", ("source", "title")),
    ("publicationVenue.name", ("publicationVenue", "name")),
    ("publicationVenue", ("publicationVenue",)),
    ("venue", ("venue",)),
    ("container-title", ("container-title",)),
    ("container_title", ("container_title",)),
    ("host_venue.display_name", ("host_venue", "display_name")),
    ("primary_location.source.display_name", ("primary_location", "source", "display_name")),
]

YEAR_CANDIDATE_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("year", ("year",)),
    ("publication_year", ("publication_year",)),
    ("publicationYear", ("publicationYear",)),
    ("cover-date-year", ("cover-date-year",)),
    ("coverDate", ("coverDate",)),
    ("publication_date", ("publication_date",)),
    ("publicationDate", ("publicationDate",)),
    ("date", ("date",)),
]

PATH_CANDIDATE_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("_xml_path", ("_xml_path",)),
    ("xml_path", ("xml_path",)),
    ("_pdf_path", ("_pdf_path",)),
    ("pdf_path", ("pdf_path",)),
    ("source_path", ("source_path",)),
    ("file_path", ("file_path",)),
    ("filepath", ("filepath",)),
    ("path", ("path",)),
    ("filename", ("filename",)),
]

SOURCE_ID_CANDIDATE_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("paperId", ("paperId",)),
    ("article_id", ("article_id",)),
    ("articleId", ("articleId",)),
    ("pii", ("pii",)),
    ("eid", ("eid",)),
    ("doi", ("doi",)),
    ("corpusId", ("corpusId",)),
    ("id", ("id",)),
]


def extract_journal_from_record(record: dict) -> tuple[str | None, str]:
    for label, path in JOURNAL_CANDIDATE_PATHS:
        title = value_to_title(nested_get(record, *path))
        if title:
            return title, f"json:{label}"
    return None, "missing"


def extract_year_from_record(record: dict) -> tuple[int | None, str]:
    for label, path in YEAR_CANDIDATE_PATHS:
        year = safe_int(nested_get(record, *path))
        if year is not None:
            return year, f"json:{label}"
    return None, "missing"


def extract_record_path(record: dict) -> tuple[str | None, str]:
    for label, path in PATH_CANDIDATE_PATHS:
        value = nested_get(record, *path)
        if value:
            s = clean_metadata_text(value)
            if s:
                return s, f"json:{label}"
    return None, "missing"


def extract_source_id(record: dict, line_no: int, source_path: Path) -> str:
    for _, path in SOURCE_ID_CANDIDATE_PATHS:
        value = nested_get(record, *path)
        if value not in (None, ""):
            return str(value)
    p, _ = extract_record_path(record)
    if p:
        return p
    return f"{source_path.name}:line_{line_no}"


# ---------------------------------------------------------------------
# Optional original-XML fallback
# ---------------------------------------------------------------------

def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def compact_tag_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


JOURNAL_TAG_PRIORITY = {
    "publicationname": 0,
    "journaltitle": 1,
    "journalname": 2,
    "srctitle": 3,
    "sourcetitle": 4,
    "periodicaltitle": 5,
    "publicationtitle": 6,
}

REFERENCE_ANCESTOR_KEYS = {
    "reference", "references", "ref", "bibreference", "bibliography",
    "bibliographysec", "bibliographysection", "referenceitem", "referenceentry",
}


def _is_under_reference_section(el: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> bool:
    cur = el
    while cur in parent_map:
        cur = parent_map[cur]
        if compact_tag_key(local_name(cur.tag)) in REFERENCE_ANCESTOR_KEYS:
            return True
    return False


def extract_xml_journal_title(xml_path: Path) -> tuple[str | None, str, list[str]]:
    """Parse metadata-like journal/publication tags; generic <title> is ignored."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        return None, f"xml_parse_error:{type(e).__name__}", []

    parent_map = {child: parent for parent in root.iter() for child in parent}
    candidates: list[tuple[int, int, str, str]] = []
    seen = set()
    order = 0

    for el in root.iter():
        if _is_under_reference_section(el, parent_map):
            continue
        tag_name = local_name(el.tag)
        key = compact_tag_key(tag_name)
        if key in JOURNAL_TAG_PRIORITY:
            text = clean_metadata_text(" ".join(el.itertext()))
            if text and len(text) <= 500:
                ident = (key, text.casefold())
                if ident not in seen:
                    seen.add(ident)
                    candidates.append((JOURNAL_TAG_PRIORITY[key], order, text, f"xml_tag:{tag_name}"))
                    order += 1
        for attr_name, attr_value in el.attrib.items():
            attr_local = local_name(attr_name)
            attr_key = compact_tag_key(attr_local)
            if attr_key in JOURNAL_TAG_PRIORITY:
                text = clean_metadata_text(attr_value)
                if text and len(text) <= 500:
                    ident = (f"attr:{attr_key}", text.casefold())
                    if ident not in seen:
                        seen.add(ident)
                        candidates.append((JOURNAL_TAG_PRIORITY[attr_key], order, text, f"xml_attr:{attr_local}"))
                        order += 1

    if not candidates:
        return None, "xml_missing", []
    candidates.sort(key=lambda x: (x[0], x[1]))
    distinct_titles: list[str] = []
    seen_titles = set()
    for _, _, title, _ in candidates:
        k = title.casefold()
        if k not in seen_titles:
            seen_titles.add(k)
            distinct_titles.append(title)
    selected = candidates[0]
    return selected[2], selected[3], distinct_titles


# ---------------------------------------------------------------------
# Path mapping / normalization / aliases / exclusions
# ---------------------------------------------------------------------

def read_path_journal_map(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, 1):
            if not row or not any(c.strip() for c in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if line_no == 1 and len(row) >= 2 and row[0].strip().casefold() == "path_substring" and row[1].strip().casefold() == "journal_title":
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_no}: expected path_substring<TAB>journal_title")
            pattern, title = row[0].strip(), row[1].strip()
            if not pattern or not title:
                raise ValueError(f"{path}:{line_no}: blank mapping field")
            rows.append((pattern, title))
    return rows


def journal_from_path(record_path: str | None, mappings: list[tuple[str, str]]) -> tuple[str | None, str]:
    if not record_path:
        return None, ""
    norm_path = str(record_path).replace("\\", "/").casefold()
    for pattern, title in mappings:
        if pattern.replace("\\", "/").casefold() in norm_path:
            return title, f"path_map:{pattern}"
    return None, ""


def normalize_journal_key(title: str | None) -> str:
    if not title:
        return ""
    s = unicodedata.normalize("NFKC", str(title)).replace("\u00a0", " ").replace("&", " and ").casefold()
    chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        chars.append(ch if cat.startswith("L") or cat.startswith("N") else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def read_aliases(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    aliases: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, 1):
            if not row or not any(c.strip() for c in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if line_no == 1 and len(row) >= 2 and row[0].strip().casefold() == "alias" and row[1].strip().casefold() == "canonical":
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_no}: expected alias<TAB>canonical")
            alias, canonical = row[0].strip(), row[1].strip()
            if not alias or not canonical:
                raise ValueError(f"{path}:{line_no}: blank alias/canonical")
            akey, ckey = normalize_journal_key(alias), normalize_journal_key(canonical)
            if not akey or not ckey:
                raise ValueError(f"{path}:{line_no}: alias/canonical normalizes to blank")
            value = (ckey, canonical)
            old = aliases.get(akey)
            if old and old != value:
                raise ValueError(f"{path}:{line_no}: conflicting mapping for {alias!r}: {old[1]!r} vs {canonical!r}")
            aliases[akey] = value
            aliases.setdefault(ckey, (ckey, canonical))
    return aliases


def resolve_alias(key: str, aliases: dict[str, tuple[str, str]]) -> tuple[str, str | None]:
    if not key:
        return "", None
    cur, display = key, None
    seen = set()
    for _ in range(20):
        if cur in seen:
            raise ValueError(f"Alias cycle detected at normalized key {cur!r}")
        seen.add(cur)
        target = aliases.get(cur)
        if target is None:
            return cur, display
        next_key, next_display = target
        display = next_display or display
        if next_key == cur:
            return cur, display
        cur = next_key
    raise ValueError(f"Alias chain too deep starting from {key!r}")


def read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    out = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = normalize_journal_key(line)
            if key:
                out.add(key)
    return out


SUSPICIOUS_NONJOURNAL_REGEXES = [
    ("arxiv", re.compile(r"\barxiv\b", re.I)),
    ("biorxiv", re.compile(r"\bbiorxiv\b", re.I)),
    ("medrxiv", re.compile(r"\bmedrxiv\b", re.I)),
    ("ssrn", re.compile(r"\bssrn\b", re.I)),
    ("research_square", re.compile(r"\bresearch\s+square\b", re.I)),
    ("zenodo", re.compile(r"\bzenodo\b", re.I)),
    ("figshare", re.compile(r"\bfigshare\b", re.I)),
    ("repository", re.compile(r"\brepositor(?:y|ies)\b", re.I)),
    ("conference_proceedings", re.compile(r"\bconference\s+proceedings\b", re.I)),
    ("proceedings_conference", re.compile(r"\bproceedings\b.*\bconference\b", re.I)),
    ("book_chapter", re.compile(r"\bbook\s+chapter\b", re.I)),
    ("thesis", re.compile(r"\bthesis\b", re.I)),
    ("dissertation", re.compile(r"\bdissertation\b", re.I)),
    ("preprint", re.compile(r"\bpreprint\b", re.I)),
    ("patent", re.compile(r"\bpatent\b", re.I)),
]


def suspicious_nonjournal_reason(title: str | None) -> str:
    if not title:
        return ""
    for label, rx in SUSPICIOUS_NONJOURNAL_REGEXES:
        if rx.search(title):
            return label
    return ""


# ---------------------------------------------------------------------
# DMJ provenance helpers
# ---------------------------------------------------------------------

def collect_pdf_basenames(folder: Path) -> set[str]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"DMJ directory not found: {folder}")
    return {p.name.casefold() for p in folder.iterdir() if p.is_file() and p.suffix.casefold() == ".pdf"}


def selected_pdf_basename(record: dict) -> str:
    p, _ = extract_record_path(record)
    if p:
        return Path(p).name.casefold()
    pid = record.get("paperId") or record.get("paper_id")
    return f"{pid}.pdf".casefold() if pid else ""


# ---------------------------------------------------------------------
# Common audit-record structure
# ---------------------------------------------------------------------

FIELDNAMES = [
    "period",
    "source_type",
    "source_file",
    "line_no",
    "source_id",
    "year",
    "year_source",
    "period_warning",
    "raw_journal_title",
    "normalized_journal_key",
    "canonical_journal_key",
    "canonical_journal_title",
    "journal_source",
    "record_path",
    "record_path_source",
    "xml_input_root_label",
    "status",
    "status_reason",
    "suspicious_nonjournal_reason",
    "xml_fallback_candidates",
    "dmj_folder_member",
]


def period_warning_for_year(year: int | None, min_year: int, max_year: int) -> str:
    if year is None:
        return "unknown_year"
    if year < min_year or year > max_year:
        return f"known_year_outside_{min_year}_{max_year}"
    return ""


# ---------------------------------------------------------------------
# XML included-manifest processing
# ---------------------------------------------------------------------

def xml_manifest_record_path(row: dict) -> tuple[str | None, str]:
    abspath = clean_metadata_text(row.get("source_abspath"))
    if abspath:
        return abspath, "manifest:source_abspath"
    root = clean_metadata_text(row.get("input_root_label"))
    rel = clean_metadata_text(row.get("source_relpath"))
    if root and rel:
        return f"{root}/{rel}", "manifest:input_root_label+source_relpath"
    if rel:
        return rel, "manifest:source_relpath"
    fn = clean_metadata_text(row.get("filename"))
    if fn:
        return fn, "manifest:filename"
    return None, "missing"


def xml_manifest_source_id(row: dict, row_no: int, manifest_path: Path) -> str:
    doi = clean_metadata_text(row.get("doi"))
    if doi:
        return doi
    pii = clean_metadata_text(row.get("pii"))
    if pii:
        return pii
    p, _ = xml_manifest_record_path(row)
    if p:
        return p
    return f"{manifest_path.name}:row_{row_no}"


def process_xml_manifest(
    *,
    manifest_path: Path,
    period: str,
    min_year: int,
    max_year: int,
    xml_fallback: bool,
    path_map: list[tuple[str, str]],
    records: list[dict],
    counters: Counter,
    xml_tag_usage: Counter,
    xml_root_counts: Counter,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"journal_title", "historical_publication_year", "input_root_label"}
        missing_cols = sorted(required - set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(f"{manifest_path}: missing required columns: {', '.join(missing_cols)}")

        for row_no, row in enumerate(reader, start=2):
            counters[f"{period}.XML.manifest_rows"] += 1

            # The canonical input should be *.included.csv. If a full manifest is
            # accidentally supplied, process only rows explicitly marked included.
            included_flag = parse_bool(row.get("historically_included"))
            selection_status = clean_metadata_text(row.get("historical_selection_status"))
            if included_flag is False or selection_status.startswith("excluded_"):
                counters[f"{period}.XML.skipped_nonincluded_manifest_rows"] += 1
                continue

            counters[f"{period}.XML.selected_source_records"] += 1

            year = safe_int(row.get("historical_publication_year"))
            year_source = "manifest:historical_publication_year" if year is not None else "missing"
            warning = period_warning_for_year(year, min_year, max_year)
            if warning:
                counters[f"{period}.XML.{warning}"] += 1

            root_label = clean_metadata_text(row.get("input_root_label"))
            xml_root_counts[(period, root_label or "[missing root label]")] += 1

            record_path, record_path_source = xml_manifest_record_path(row)
            source_id = xml_manifest_source_id(row, row_no, manifest_path)

            journal = clean_metadata_text(row.get("journal_title")) or None
            journal_source = "manifest:journal_title" if journal else "missing"
            fallback_candidates: list[str] = []

            # Optional raw-XML fallback only when source_abspath is present and valid.
            if not journal and xml_fallback:
                abspath = clean_metadata_text(row.get("source_abspath"))
                p = Path(abspath) if abspath else None
                if p and p.exists() and p.is_file() and p.suffix.casefold() == ".xml":
                    fallback, provenance, candidates = extract_xml_journal_title(p)
                    fallback_candidates = candidates
                    if fallback:
                        journal = fallback
                        journal_source = provenance
                        counters[f"{period}.XML.journal_from_raw_xml"] += 1
                        xml_tag_usage[(period, provenance)] += 1
                else:
                    counters[f"{period}.XML.raw_xml_path_unavailable"] += 1

            if not journal:
                fallback, provenance = journal_from_path(record_path, path_map)
                if fallback:
                    journal = fallback
                    journal_source = provenance
                    counters[f"{period}.XML.journal_from_path_map"] += 1

            if journal:
                counters[f"{period}.XML.journal_present"] += 1
            else:
                counters[f"{period}.XML.journal_missing"] += 1

            records.append({
                "period": period,
                "source_type": "XML",
                "source_file": str(manifest_path),
                "line_no": row_no,
                "source_id": source_id,
                "year": "" if year is None else year,
                "year_source": year_source,
                "period_warning": warning,
                "raw_journal_title": journal or "",
                "normalized_journal_key": normalize_journal_key(journal),
                "canonical_journal_key": "",
                "canonical_journal_title": "",
                "journal_source": journal_source,
                "record_path": record_path or "",
                "record_path_source": record_path_source,
                "xml_input_root_label": root_label,
                "status": "pending",
                "status_reason": "",
                "suspicious_nonjournal_reason": suspicious_nonjournal_reason(journal),
                "xml_fallback_candidates": " || ".join(fallback_candidates),
                "dmj_folder_member": "",
            })


# ---------------------------------------------------------------------
# PDF selected-record JSONL processing
# ---------------------------------------------------------------------

def process_pdf_jsonl(
    *,
    jsonl_path: Path,
    period: str,
    min_year: int,
    max_year: int,
    dmj_basenames: set[str],
    path_map: list[tuple[str, str]],
    records: list[dict],
    counters: Counter,
    dmj_audit_rows: list[dict],
) -> None:
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    selected_dmj_names: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            counters[f"{period}.PDF.jsonl_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                counters[f"{period}.PDF.bad_json"] += 1
                records.append({
                    "period": period, "source_type": "PDF", "source_file": str(jsonl_path),
                    "line_no": line_no, "source_id": f"{jsonl_path.name}:line_{line_no}",
                    "year": "", "year_source": "", "period_warning": "",
                    "raw_journal_title": "", "normalized_journal_key": "",
                    "canonical_journal_key": "", "canonical_journal_title": "",
                    "journal_source": "", "record_path": "", "record_path_source": "",
                    "xml_input_root_label": "", "status": "bad_json", "status_reason": str(e),
                    "suspicious_nonjournal_reason": "", "xml_fallback_candidates": "",
                    "dmj_folder_member": "",
                })
                continue
            if not isinstance(record, dict):
                counters[f"{period}.PDF.non_object_json"] += 1
                continue
            if "_metadata" in record and len(record) == 1:
                counters[f"{period}.PDF.metadata_rows_skipped"] += 1
                continue

            counters[f"{period}.PDF.selected_source_records"] += 1
            source_id = extract_source_id(record, line_no, jsonl_path)
            year, year_source = extract_year_from_record(record)
            warning = period_warning_for_year(year, min_year, max_year)
            if warning:
                counters[f"{period}.PDF.{warning}"] += 1

            record_path, record_path_source = extract_record_path(record)
            basename = selected_pdf_basename(record)
            is_dmj_member = bool(basename and basename in dmj_basenames)
            if is_dmj_member:
                selected_dmj_names.add(basename)
                counters[f"{period}.PDF.selected_records_in_dmj_folder"] += 1

            journal, journal_source = extract_journal_from_record(record)

            # Historical extra-folder PDF records can carry the sentinel source label
            # ``EXTRA_PDF_DIR`` in the journal field.  It is not a journal title.
            # Stage 1B independently established that selected records whose exact PDF
            # filenames occur in the period-specific DMJ extra folders belong to
            # Dental Materials Journal.  Therefore the DMJ provenance rule applies both
            # when journal metadata are blank and when this exact historical sentinel is
            # present.  We deliberately do NOT reinterpret any other nonblank title.
            journal_key = normalize_journal_key(journal) if journal else ""
            is_extra_pdf_dir_placeholder = journal_key == "extra pdf dir"

            if is_dmj_member and (not journal or is_extra_pdf_dir_placeholder):
                if is_extra_pdf_dir_placeholder:
                    counters[f"{period}.PDF.extra_pdf_dir_placeholder_reassigned_to_dmj"] += 1
                    journal_source = "dmj_folder_provenance_from_EXTRA_PDF_DIR"
                else:
                    journal_source = "dmj_folder_provenance_from_blank_journal"
                journal = "Dental Materials Journal"
                counters[f"{period}.PDF.journal_from_dmj_folder_provenance"] += 1

            if not journal:
                fallback, provenance = journal_from_path(record_path, path_map)
                if fallback:
                    journal = fallback
                    journal_source = provenance
                    counters[f"{period}.PDF.journal_from_path_map"] += 1

            if journal:
                counters[f"{period}.PDF.journal_present"] += 1
            else:
                counters[f"{period}.PDF.journal_missing"] += 1

            records.append({
                "period": period,
                "source_type": "PDF",
                "source_file": str(jsonl_path),
                "line_no": line_no,
                "source_id": source_id,
                "year": "" if year is None else year,
                "year_source": year_source,
                "period_warning": warning,
                "raw_journal_title": journal or "",
                "normalized_journal_key": normalize_journal_key(journal),
                "canonical_journal_key": "",
                "canonical_journal_title": "",
                "journal_source": journal_source,
                "record_path": record_path or "",
                "record_path_source": record_path_source,
                "xml_input_root_label": "",
                "status": "pending",
                "status_reason": "",
                "suspicious_nonjournal_reason": suspicious_nonjournal_reason(journal),
                "xml_fallback_candidates": "",
                "dmj_folder_member": "yes" if is_dmj_member else "no",
            })

    dmj_only = sorted(dmj_basenames - selected_dmj_names)
    selected_count = len(selected_dmj_names)
    counters[f"{period}.PDF.dmj_folder_pdf_files"] = len(dmj_basenames)
    counters[f"{period}.PDF.dmj_folder_files_not_in_selected_records"] = len(dmj_only)
    dmj_audit_rows.append({
        "period": period,
        "dmj_folder_pdf_files": len(dmj_basenames),
        "selected_pdf_records_matching_dmj_folder": selected_count,
        "dmj_folder_files_not_in_selected_records": len(dmj_only),
        "journal_filled_from_dmj_folder_provenance": counters[f"{period}.PDF.journal_from_dmj_folder_provenance"],
        "unmatched_dmj_filenames": " || ".join(dmj_only),
    })


# ---------------------------------------------------------------------
# Canonicalization and summary construction
# ---------------------------------------------------------------------

def choose_preferred_raw_title(counter: Counter) -> str:
    if not counter:
        return ""
    return sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0].casefold()))[0][0]


def finalize_records(records: list[dict], aliases: dict[str, tuple[str, str]], exclusions: set[str]) -> None:
    raw_by_key: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r["status"] != "pending":
            continue
        raw, key = r["raw_journal_title"], r["normalized_journal_key"]
        if raw and key:
            raw_by_key[key][raw] += 1

    resolution = {key: resolve_alias(key, aliases) for key in raw_by_key}
    raw_by_final_key: dict[str, Counter] = defaultdict(Counter)
    alias_display: dict[str, str] = {}
    for raw_key, raw_counter in raw_by_key.items():
        final_key, preferred = resolution[raw_key]
        raw_by_final_key[final_key].update(raw_counter)
        if preferred:
            alias_display.setdefault(final_key, preferred)

    final_display = {
        final_key: alias_display.get(final_key) or choose_preferred_raw_title(raw_counter)
        for final_key, raw_counter in raw_by_final_key.items()
    }

    for r in records:
        if r["status"] != "pending":
            continue
        raw_key = r["normalized_journal_key"]
        if not raw_key:
            r["status"] = "excluded_from_journal_stats"
            r["status_reason"] = "missing_journal_title"
            continue
        final_key, preferred = resolution.get(raw_key, resolve_alias(raw_key, aliases))
        r["canonical_journal_key"] = final_key
        r["canonical_journal_title"] = preferred or final_display.get(final_key) or r["raw_journal_title"]
        if raw_key in exclusions or final_key in exclusions:
            r["status"] = "excluded_from_journal_stats"
            r["status_reason"] = "explicit_exclusion"
        else:
            r["status"] = "included_in_journal_stats"


def build_journal_counts(records: list[dict]) -> list[dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    display: dict[str, str] = {}
    for r in records:
        if r["status"] != "included_in_journal_stats":
            continue
        key = r["canonical_journal_key"]
        display.setdefault(key, r["canonical_journal_title"])
        counts[key][r["period"]] += 1
        counts[key][f"{r['period']}_{r['source_type']}"] += 1

    rows = []
    for key, c in counts.items():
        pre_xml, pre_pdf = c["pre2018_XML"], c["pre2018_PDF"]
        post_xml, post_pdf = c["post2018_XML"], c["post2018_PDF"]
        pre, post = pre_xml + pre_pdf, post_xml + post_pdf
        rows.append({
            "journal": display[key],
            "pre2018_xml": pre_xml,
            "pre2018_pdf": pre_pdf,
            "pre2018_records": pre,
            "post2018_xml": post_xml,
            "post2018_pdf": post_pdf,
            "post2018_records": post,
            "total_records": pre + post,
            "normalized_key": key,
        })
    rows.sort(key=lambda r: (-r["total_records"], r["journal"].casefold()))
    return rows


def build_mapping_rows(records: list[dict]) -> list[dict]:
    agg: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for r in records:
        raw = r["raw_journal_title"]
        if not raw:
            continue
        key = (raw, r["normalized_journal_key"], r["canonical_journal_key"], r["canonical_journal_title"])
        agg[key][r["period"]] += 1
        agg[key][f"{r['period']}_{r['source_type']}"] += 1
        if r["status"] == "excluded_from_journal_stats":
            agg[key]["excluded"] += 1
        if r["suspicious_nonjournal_reason"]:
            agg[key][f"suspicious:{r['suspicious_nonjournal_reason']}"] += 1

    rows = []
    for key, c in agg.items():
        raw, norm_key, canon_key, canon_title = key
        suspicious = sorted(k.split(":", 1)[1] for k in c if k.startswith("suspicious:"))
        rows.append({
            "raw_journal_title": raw,
            "normalized_journal_key": norm_key,
            "canonical_journal_key": canon_key,
            "canonical_journal_title": canon_title,
            "pre2018_xml": c["pre2018_XML"],
            "pre2018_pdf": c["pre2018_PDF"],
            "post2018_xml": c["post2018_XML"],
            "post2018_pdf": c["post2018_PDF"],
            "pre2018_records": c["pre2018"],
            "post2018_records": c["post2018"],
            "total_records": c["pre2018"] + c["post2018"],
            "excluded_records": c["excluded"],
            "suspicious_nonjournal_reason": ";".join(suspicious),
        })
    rows.sort(key=lambda r: (-r["total_records"], r["raw_journal_title"].casefold()))
    return rows


def build_review_candidates(records: list[dict]) -> list[dict]:
    agg: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        raw = r["raw_journal_title"]
        reasons = []
        if r["suspicious_nonjournal_reason"]:
            reasons.append(r["suspicious_nonjournal_reason"])
        if r["xml_fallback_candidates"] and " || " in r["xml_fallback_candidates"]:
            reasons.append("multiple_xml_journal_candidates")
        for reason in reasons:
            agg[(raw, reason)][r["period"]] += 1
            agg[(raw, reason)][f"{r['period']}_{r['source_type']}"] += 1

    rows = []
    for (raw, reason), c in agg.items():
        rows.append({
            "raw_journal_title": raw,
            "reason": reason,
            "pre2018_xml": c["pre2018_XML"],
            "pre2018_pdf": c["pre2018_PDF"],
            "post2018_xml": c["post2018_XML"],
            "post2018_pdf": c["post2018_PDF"],
            "pre2018_records": c["pre2018"],
            "post2018_records": c["post2018"],
            "total_records": c["pre2018"] + c["post2018"],
        })
    rows.sort(key=lambda r: (-r["total_records"], r["raw_journal_title"].casefold()))
    return rows


def build_panel_a(journal_counts: list[dict], min_total: int) -> list[dict]:
    above = [dict(r) for r in journal_counts if r["total_records"] >= min_total]
    below = [r for r in journal_counts if r["total_records"] < min_total]
    if below:
        other = {
            "journal": f"Other (<{min_total} records each)",
            "pre2018_xml": sum(r["pre2018_xml"] for r in below),
            "pre2018_pdf": sum(r["pre2018_pdf"] for r in below),
            "pre2018_records": sum(r["pre2018_records"] for r in below),
            "post2018_xml": sum(r["post2018_xml"] for r in below),
            "post2018_pdf": sum(r["post2018_pdf"] for r in below),
            "post2018_records": sum(r["post2018_records"] for r in below),
            "total_records": sum(r["total_records"] for r in below),
            "normalized_key": "[other_below_threshold]",
        }
        above.append(other)
    return above


def read_panel_b_titles(path: Path | None) -> list[str]:
    if path is None:
        return []
    out = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def build_panel_b(journal_counts: list[dict], titles: list[str], aliases: dict[str, tuple[str, str]], warnings: list[str]) -> list[dict]:
    by_key = {r["normalized_key"]: r for r in journal_counts}
    rows = []
    seen = set()
    for requested in titles:
        raw_key = normalize_journal_key(requested)
        final_key, preferred = resolve_alias(raw_key, aliases)
        if final_key in seen:
            continue
        seen.add(final_key)
        found = by_key.get(final_key)
        if found:
            rows.append(dict(found))
        else:
            warnings.append(f"Panel B requested journal not found after canonicalization: {requested}")
            rows.append({
                "journal": preferred or requested,
                "pre2018_xml": 0, "pre2018_pdf": 0, "pre2018_records": 0,
                "post2018_xml": 0, "post2018_pdf": 0, "post2018_records": 0,
                "total_records": 0, "normalized_key": final_key,
            })
    return rows


# ---------------------------------------------------------------------
# Templates / checks
# ---------------------------------------------------------------------

def write_templates(out_dir: Path) -> None:
    p = out_dir / "journal_aliases_template.tsv"
    if not p.exists():
        p.write_text(
            "alias\tcanonical\n"
            "# Review journal_title_mapping.csv before adding mappings.\n"
            "# Example:\n"
            "# Dental materials : official publication of the Academy of Dental Materials\tDental Materials\n",
            encoding="utf-8",
        )

    p = out_dir / "journal_exclusions_template.txt"
    if not p.exists():
        p.write_text(
            "# One explicitly excluded non-journal source title per line.\n"
            "# Review journal_review_candidates.csv and journal_title_mapping.csv first.\n",
            encoding="utf-8",
        )

    p = out_dir / "path_journal_map_template.tsv"
    if not p.exists():
        p.write_text(
            "path_substring\tjournal_title\n"
            "# Last-resort mapping only when metadata/provenance did not resolve journal title.\n",
            encoding="utf-8",
        )

    p = out_dir / "panel_b_journals_template.txt"
    if not p.exists():
        p.write_text(
            "# One canonical selected dental/restorative journal per line.\n"
            "# Example:\n"
            "# Dental Materials\n"
            "# Journal of Dentistry\n"
            "# The Journal of Prosthetic Dentistry\n"
            "# Dental Materials Journal\n",
            encoding="utf-8",
        )


def check_expected(label: str, actual: int, expected: int | None, warnings: list[str]) -> None:
    if expected is None or expected <= 0:
        return
    if actual != expected:
        warnings.append(f"{label}: expected {expected:,}, observed {actual:,}")


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical journal statistics audit from XML included manifests + selected PDF JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pre-xml-manifest", type=Path, required=True)
    p.add_argument("--post-xml-manifest", type=Path, required=True)
    p.add_argument("--pre-pdf-jsonl", type=Path, required=True)
    p.add_argument("--post-pdf-jsonl", type=Path, required=True)
    p.add_argument("--pre-dmj-dir", type=Path, required=True)
    p.add_argument("--post-dmj-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs_journal_audit"))

    p.add_argument("--pre-min-year", type=int, default=1992)
    p.add_argument("--pre-max-year", type=int, default=2017)
    p.add_argument("--post-min-year", type=int, default=2018)
    p.add_argument("--post-max-year", type=int, default=2026)

    p.add_argument("--xml-fallback-to-files", action="store_true")
    p.add_argument("--aliases", type=Path, default=None, help="TSV: alias<TAB>canonical")
    p.add_argument("--exclude-titles", type=Path, default=None, help="One explicitly excluded non-journal title per line")
    p.add_argument("--path-journal-map", type=Path, default=None, help="TSV: path_substring<TAB>journal_title")
    p.add_argument("--panel-b-journals", type=Path, default=None, help="One selected Panel B journal per line")
    p.add_argument("--panel-a-min-total", type=int, default=500)

    # Audited source-record totals. Set any to 0 to disable that check.
    p.add_argument("--expected-pre-xml", type=int, default=50224)
    p.add_argument("--expected-post-xml", type=int, default=36988)
    p.add_argument("--expected-pre-pdf", type=int, default=45926)
    p.add_argument("--expected-post-pdf", type=int, default=32207)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    aliases = read_aliases(args.aliases)
    exclusions = read_exclusions(args.exclude_titles)
    path_map = read_path_journal_map(args.path_journal_map)
    pre_dmj = collect_pdf_basenames(args.pre_dmj_dir)
    post_dmj = collect_pdf_basenames(args.post_dmj_dir)

    records: list[dict] = []
    counters = Counter()
    xml_tag_usage = Counter()
    xml_root_counts = Counter()
    dmj_audit_rows: list[dict] = []

    process_xml_manifest(
        manifest_path=args.pre_xml_manifest, period="pre2018",
        min_year=args.pre_min_year, max_year=args.pre_max_year,
        xml_fallback=args.xml_fallback_to_files, path_map=path_map,
        records=records, counters=counters, xml_tag_usage=xml_tag_usage,
        xml_root_counts=xml_root_counts,
    )
    process_xml_manifest(
        manifest_path=args.post_xml_manifest, period="post2018",
        min_year=args.post_min_year, max_year=args.post_max_year,
        xml_fallback=args.xml_fallback_to_files, path_map=path_map,
        records=records, counters=counters, xml_tag_usage=xml_tag_usage,
        xml_root_counts=xml_root_counts,
    )
    process_pdf_jsonl(
        jsonl_path=args.pre_pdf_jsonl, period="pre2018",
        min_year=args.pre_min_year, max_year=args.pre_max_year,
        dmj_basenames=pre_dmj, path_map=path_map,
        records=records, counters=counters, dmj_audit_rows=dmj_audit_rows,
    )
    process_pdf_jsonl(
        jsonl_path=args.post_pdf_jsonl, period="post2018",
        min_year=args.post_min_year, max_year=args.post_max_year,
        dmj_basenames=post_dmj, path_map=path_map,
        records=records, counters=counters, dmj_audit_rows=dmj_audit_rows,
    )

    finalize_records(records, aliases, exclusions)
    journal_counts = build_journal_counts(records)
    mapping_rows = build_mapping_rows(records)
    review_rows = build_review_candidates(records)

    warnings: list[str] = []
    pre_xml_n = counters["pre2018.XML.selected_source_records"]
    post_xml_n = counters["post2018.XML.selected_source_records"]
    pre_pdf_n = counters["pre2018.PDF.selected_source_records"]
    post_pdf_n = counters["post2018.PDF.selected_source_records"]
    check_expected("pre2018 XML included-manifest records", pre_xml_n, args.expected_pre_xml, warnings)
    check_expected("post2018 XML included-manifest records", post_xml_n, args.expected_post_xml, warnings)
    check_expected("pre2018 PDF selected records", pre_pdf_n, args.expected_pre_pdf, warnings)
    check_expected("post2018 PDF selected records", post_pdf_n, args.expected_post_pdf, warnings)

    for period in ("pre2018", "post2018"):
        for stype in ("XML", "PDF"):
            missing_n = counters[f"{period}.{stype}.journal_missing"]
            if missing_n:
                warnings.append(f"{period} {stype}: {missing_n:,} records have no journal title after all enabled fallbacks.")

    period_warning_count = sum(1 for r in records if r.get("period_warning"))
    if period_warning_count:
        warnings.append(f"{period_warning_count:,} source records have unknown year or a known year outside the declared period; see journal_record_audit.csv. Unknown-year DMJ extras are expected from Stage 1B provenance.")

    if review_rows:
        warnings.append(f"{len(review_rows):,} raw-title/reason combinations were flagged for manual non-journal/XML-candidate review. They remain included unless explicitly listed in --exclude-titles.")

    included = [r for r in records if r["status"] == "included_in_journal_stats"]
    excluded = [r for r in records if r["status"] == "excluded_from_journal_stats"]
    missing_rows = [r for r in records if r["status"] == "excluded_from_journal_stats" and r["status_reason"] == "missing_journal_title"]

    raw_nonblank = {r["raw_journal_title"] for r in records if r["raw_journal_title"]}
    normalized_nonblank = {r["normalized_journal_key"] for r in records if r["normalized_journal_key"]}
    final_keys = {r["canonical_journal_key"] for r in included if r["canonical_journal_key"]}

    panel_a = build_panel_a(journal_counts, args.panel_a_min_total)
    panel_b_titles = read_panel_b_titles(args.panel_b_journals)
    panel_b = build_panel_b(journal_counts, panel_b_titles, aliases, warnings) if panel_b_titles else []

    common_fields = [
        "journal", "pre2018_xml", "pre2018_pdf", "pre2018_records",
        "post2018_xml", "post2018_pdf", "post2018_records", "total_records", "normalized_key",
    ]
    write_csv(args.out_dir / "journal_counts.csv", journal_counts, common_fields)
    write_csv(args.out_dir / "table_s2_panel_a.csv", panel_a, common_fields)
    if panel_b_titles:
        write_csv(args.out_dir / "table_s2_panel_b.csv", panel_b, common_fields)
    write_csv(args.out_dir / "journal_title_mapping.csv", mapping_rows)
    write_csv(args.out_dir / "journal_record_audit.csv", records, FIELDNAMES)
    write_csv(args.out_dir / "missing_journal_records.csv", missing_rows, FIELDNAMES)
    write_csv(args.out_dir / "journal_review_candidates.csv", review_rows)

    xml_tag_rows = [
        {"period": period, "xml_journal_source": provenance, "count": count}
        for (period, provenance), count in xml_tag_usage.items()
    ]
    xml_tag_rows.sort(key=lambda r: (r["period"], -r["count"], r["xml_journal_source"]))
    write_csv(args.out_dir / "xml_fallback_tag_usage.csv", xml_tag_rows, ["period", "xml_journal_source", "count"])

    root_rows = [
        {"period": period, "input_root_label": root, "count": count}
        for (period, root), count in xml_root_counts.items()
    ]
    root_rows.sort(key=lambda r: (r["period"], -r["count"], r["input_root_label"]))
    write_csv(args.out_dir / "xml_root_counts.csv", root_rows, ["period", "input_root_label", "count"])
    write_csv(
        args.out_dir / "dmj_provenance_audit.csv",
        dmj_audit_rows,
        ["period", "dmj_folder_pdf_files", "selected_pdf_records_matching_dmj_folder", "dmj_folder_files_not_in_selected_records", "journal_filled_from_dmj_folder_provenance", "unmatched_dmj_filenames"],
    )

    summary = {
        "settings": {
            "pre_xml_manifest": str(args.pre_xml_manifest),
            "post_xml_manifest": str(args.post_xml_manifest),
            "pre_pdf_jsonl": str(args.pre_pdf_jsonl),
            "post_pdf_jsonl": str(args.post_pdf_jsonl),
            "pre_dmj_dir": str(args.pre_dmj_dir),
            "post_dmj_dir": str(args.post_dmj_dir),
            "pre_period": [args.pre_min_year, args.pre_max_year],
            "post_period": [args.post_min_year, args.post_max_year],
            "xml_fallback_to_files": args.xml_fallback_to_files,
            "aliases_file": str(args.aliases) if args.aliases else None,
            "exclude_titles_file": str(args.exclude_titles) if args.exclude_titles else None,
            "path_journal_map_file": str(args.path_journal_map) if args.path_journal_map else None,
            "panel_b_journals_file": str(args.panel_b_journals) if args.panel_b_journals else None,
            "panel_a_min_total": args.panel_a_min_total,
        },
        "source_record_counts": {
            "pre2018": {"xml": pre_xml_n, "pdf": pre_pdf_n, "combined": pre_xml_n + pre_pdf_n},
            "post2018": {"xml": post_xml_n, "pdf": post_pdf_n, "combined": post_xml_n + post_pdf_n},
        },
        "counters": dict(sorted(counters.items())),
        "journal_statistics": {
            "unique_raw_nonblank_labels": len(raw_nonblank),
            "unique_base_normalized_nonblank_labels": len(normalized_nonblank),
            "unique_final_included_journals": len(final_keys),
            "included_source_records": len(included),
            "excluded_source_records": len(excluded),
            "missing_journal_records": len(missing_rows),
            "panel_a_named_journals": sum(1 for r in panel_a if r["normalized_key"] != "[other_below_threshold]"),
            "panel_b_journals": len(panel_b),
        },
        "dmj_provenance": dmj_audit_rows,
        "warnings": warnings,
        "notes": [
            "Each included XML-manifest row and selected PDF JSONL row is one source record.",
            "No new DOI/title bibliographic deduplication is performed.",
            "Exact-PDF file deduplication and XML/PDF cross-source deduplication were audited separately in Stage 1C.",
            "Year is used only for audit warnings here; source records are not re-filtered.",
            "Missing PDF journal metadata is assigned to Dental Materials Journal only by exact filename membership in the audited period-specific DMJ folder.",
            "Suspicious non-journal labels are review flags only and are not automatically excluded.",
            "Known journal-title variants should be supplied explicitly in the alias TSV.",
            "Counts describe source records and may not equal deduplicated unique publications.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_templates(args.out_dir)

    print("\n=== CANONICAL JOURNAL AUDIT SUMMARY ===")
    print(f"Pre XML included records:   {pre_xml_n:,}")
    print(f"Post XML included records:  {post_xml_n:,}")
    print(f"Pre PDF selected records:   {pre_pdf_n:,}")
    print(f"Post PDF selected records:  {post_pdf_n:,}")
    print(f"Pre combined source records:{pre_xml_n + pre_pdf_n:>11,}")
    print(f"Post combined source records:{post_xml_n + post_pdf_n:>10,}")
    print(f"Raw nonblank journal labels:         {len(raw_nonblank):,}")
    print(f"Base-normalized nonblank labels:     {len(normalized_nonblank):,}")
    print(f"Final included normalized journals:  {len(final_keys):,}")
    print(f"Included source records in stats:    {len(included):,}")
    print(f"Missing journal-title records:       {len(missing_rows):,}")
    print(f"Panel A named journals (>= {args.panel_a_min_total}): {sum(1 for r in panel_a if r['normalized_key'] != '[other_below_threshold]'):,}")
    print(f"Pre DMJ folder PDFs: {len(pre_dmj):,}; selected matches: {counters['pre2018.PDF.selected_records_in_dmj_folder']:,}; provenance fills: {counters['pre2018.PDF.journal_from_dmj_folder_provenance']:,}; EXTRA_PDF_DIR reassigned: {counters['pre2018.PDF.extra_pdf_dir_placeholder_reassigned_to_dmj']:,}")
    print(f"Post DMJ folder PDFs: {len(post_dmj):,}; selected matches: {counters['post2018.PDF.selected_records_in_dmj_folder']:,}; provenance fills: {counters['post2018.PDF.journal_from_dmj_folder_provenance']:,}; EXTRA_PDF_DIR reassigned: {counters['post2018.PDF.extra_pdf_dir_placeholder_reassigned_to_dmj']:,}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    print(f"\nOutputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
