from __future__ import annotations

import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import PipelinePaths
from .io_utils import iter_jsonl, stable_id, write_jsonl
from .pdf_extraction import extract_pdf_text
from .reporting import write_stage_report
from .xml_extraction import extract_elsevier_xml_texts


def _passes_year_filter(year: int | None, config: dict[str, Any]) -> bool:
    extraction = config.get("extraction", {})
    if year is None:
        return not bool(extraction.get("skip_unknown_year", False))
    min_year = extraction.get("min_year")
    max_year = extraction.get("max_year")
    return (min_year is None or year >= int(min_year)) and (
        max_year is None or year <= int(max_year)
    )


def _extract_record(
    manifest_record: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    extraction = config.get("extraction", {})
    min_chars = int(extraction.get("min_chars", 30))
    source_type = manifest_record["source_type"]
    ingest_type = manifest_record.get("ingest_type", source_type)
    year = manifest_record.get("publication_year")

    if ingest_type == "inline":
        chunks = [manifest_record["inline_text"]]
    elif ingest_type == "pdf":
        chunks = extract_pdf_text(
            manifest_record["source_path"],
            min_chars=min_chars,
            repair_after_eof=bool(extraction.get("repair_pdf_after_eof", True)),
        )
    elif ingest_type == "xml":
        chunks, xml_year = extract_elsevier_xml_texts(
            manifest_record["source_path"],
            split=extraction.get("xml_split", "section"),
            include_title=bool(extraction.get("include_title", True)),
            include_abstract=bool(extraction.get("include_abstract", True)),
            include_keywords=bool(extraction.get("include_keywords", True)),
            include_section_titles=bool(
                extraction.get("include_section_titles", True)
            ),
            min_chars=min_chars,
        )
        year = year or xml_year
    else:
        raise ValueError(f"Unsupported ingest type: {ingest_type}")

    if not _passes_year_filter(year, config):
        return [], "filtered_by_year"

    rows: list[dict[str, Any]] = []
    for chunk_index, text in enumerate(chunks):
        text = text.strip()
        if len(text) < min_chars:
            continue
        rows.append(
            {
                "chunk_id": stable_id(manifest_record["document_id"], chunk_index),
                "document_id": manifest_record["document_id"],
                "period": manifest_record["period"],
                "source_type": source_type,
                "source_path": manifest_record["source_path"],
                "publication_year": year,
                "chunk_index": chunk_index,
                "text": text,
            }
        )
    return rows, None


def _iter_extracted(
    config: dict[str, Any],
    paths: PipelinePaths,
    counters: Counter[str],
    errors: list[dict[str, str]],
) -> Iterable[dict[str, Any]]:
    for manifest_record in iter_jsonl(paths.manifest_path):
        if not manifest_record.get("selected"):
            continue
        counters["selected_documents"] += 1
        try:
            rows, status = _extract_record(manifest_record, config)
            if status:
                counters[status] += 1
            if not rows:
                counters["documents_without_text"] += 1
            else:
                counters["documents_with_text"] += 1
            for row in rows:
                counters["chunks"] += 1
                counters[f"chunks_{row['period']}"] += 1
                counters["characters"] += len(row["text"])
                yield row
        except Exception as exc:  # stage continues and exposes errors
            counters["failed_documents"] += 1
            errors.append(
                {
                    "source_path": manifest_record["source_path"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    n_rows = write_jsonl(
        _iter_extracted(config, paths, counters, errors),
        paths.raw_documents_path,
    )
    error_path = paths.raw_documents_path.parent / "extraction_errors.jsonl"
    write_jsonl(errors, error_path)

    metrics = dict(counters)
    metrics["written_chunks"] = n_rows
    metrics["n_errors"] = len(errors)
    write_stage_report(
        report_path=paths.reports_dir / "02_extraction.md",
        title="Stage 2 — PDF/XML extraction",
        purpose=(
            "Extracts text from PDF and Elsevier XML sources into a shared JSONL "
            "schema while preserving source provenance."
        ),
        inputs=[paths.manifest_path],
        outputs=[paths.raw_documents_path, error_path],
        metrics=metrics,
        parameters=config.get("extraction", {}),
        notes=[
            "A single source document may produce multiple chunks (for example XML sections).",
            "Failures are retained in extraction_errors.jsonl rather than silently discarded.",
        ],
    )
    return metrics
