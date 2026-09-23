#!/usr/bin/env python3
"""
Compare PyMuPDF and pypdf text extraction quality on a PDF corpus.

This script is intentionally independent from the main Word2Vec pipeline.
It does not modify your corpus. It only:
  1. collects PDF files from one or more folders;
  2. extracts text with PyMuPDF and pypdf;
  3. computes quality metrics and domain-term counts;
  4. writes CSV summaries for manual inspection.

Install dependencies:
    pip install pymupdf pypdf

Example:
    python compare_pdf_extractors.py \
      --input-dirs data/dental_pre2018 data/dental_post2018 \
      --sample-size 200 \
      --seed 42 \
      --out-dir outputs/pdf_extraction_benchmark \
      --save-text-samples 10

Outputs:
    extraction_metrics_long.csv
        One row per PDF per backend.

    extraction_metrics_wide.csv
        One row per PDF with pypdf/PyMuPDF metrics side by side.

    summary_by_backend.csv
        Aggregate statistics per backend.

    term_counts_by_backend.csv
        Aggregate counts for important scientific/materials terms.

    text_samples/
        Optional raw extracted text samples for side-by-side inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import io
import logging
import warnings
from typing import Any


DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
    }
)

# Explicit mappings are kept even though NFKC already handles many of them.
# This makes the intended scientific normalization visible and testable.
SUBSCRIPT_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
        "₊": "+",
        "₋": "-",
    }
)


TERM_PATTERNS = {
    "3Y-TZP": r"\b3\s*Y\s*-\s*TZP\b",
    "4Y-TZP": r"\b4\s*Y\s*-\s*TZP\b",
    "5Y-TZP": r"\b5\s*Y\s*-\s*TZP\b",
    "3Y-PSZ": r"\b3\s*Y\s*-\s*PSZ\b",
    "4Y-PSZ": r"\b4\s*Y\s*-\s*PSZ\b",
    "5Y-PSZ": r"\b5\s*Y\s*-\s*PSZ\b",
    "Y-TZP": r"\bY\s*-\s*TZP\b",
    "Y-PSZ": r"\bY\s*-\s*PSZ\b",
    "H2O": r"\bH\s*2\s*O\b",
    "Ca(OH)2": r"\bCa\s*\(\s*OH\s*\)\s*2\b",
    "zirconia": r"\bzirconia\b",
    "yttria": r"\byttria\b",
    "translucency": r"\btranslucenc(?:y|ies)\b",
    "flexural_strength": r"\bflexural\s+strength\b",
    "fracture_toughness": r"\bfracture\s+toughness\b",
}


def extract_page_text_pymupdf_columns(page: Any) -> str:
    """
    Extract text from a PyMuPDF page using a simple column-aware reading order.

    The function is designed for scientific PDFs with one- or two-column layouts.
    It uses text block coordinates from page.get_text("blocks") and reconstructs
    reading order as:
      - full-width blocks top-to-bottom;
      - within each region between full-width blocks:
          left column top-to-bottom, then right column top-to-bottom.
    """
    page_rect = page.rect
    page_width = page_rect.width
    mid_x = (page_rect.x0 + page_rect.x1) / 2

    raw_blocks = page.get_text("blocks", sort=False)

    blocks = []
    for block in raw_blocks:
        # PyMuPDF block tuple is usually:
        # (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        block_type = block[6] if len(block) > 6 else 0

        # block_type 0 = text, 1 = image
        if block_type != 0:
            continue

        text = (text or "").strip()
        if not text:
            continue

        width = x1 - x0
        x_center = (x0 + x1) / 2

        blocks.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": width,
                "x_center": x_center,
                "text": text,
            }
        )

    if not blocks:
        return ""

    # Full-width blocks: title, abstract header, large headings,
    # sometimes figure/table captions.
    full_width_blocks = [
        b for b in blocks
        if b["width"] >= 0.65 * page_width
    ]

    column_blocks = [
        b for b in blocks
        if b["width"] < 0.65 * page_width
    ]

    full_width_blocks = sorted(
        full_width_blocks,
        key=lambda b: (b["y0"], b["x0"])
    )

    def emit_column_region(region_blocks: list[dict]) -> list[str]:
        """
        Read a two-column region: left column top-to-bottom,
        then right column top-to-bottom.
        """
        if not region_blocks:
            return []

        left = [
            b for b in region_blocks
            if b["x_center"] < mid_x
        ]
        right = [
            b for b in region_blocks
            if b["x_center"] >= mid_x
        ]

        left = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right = sorted(right, key=lambda b: (b["y0"], b["x0"]))

        return [b["text"] for b in left + right]

    output_parts = []
    last_y = page_rect.y0

    for full_block in full_width_blocks:
        # Emit column blocks above this full-width block.
        region = [
            b for b in column_blocks
            if last_y <= b["y0"] < full_block["y0"]
        ]
        output_parts.extend(emit_column_region(region))

        # Then emit the full-width block itself.
        output_parts.append(full_block["text"])
        last_y = max(last_y, full_block["y1"])

    # Emit remaining column blocks below the last full-width block.
    region = [
        b for b in column_blocks
        if b["y0"] >= last_y
    ]
    output_parts.extend(emit_column_region(region))

    return "\n\n".join(output_parts)
    

@dataclass
class ExtractionResult:
    backend: str
    status: str
    text: str
    page_texts: list[str]
    elapsed_seconds: float
    error: str = ""


def normalize_scientific_text(text: str) -> str:
    """
    Normalize Unicode variants relevant for scientific PDF extraction.

    This is used only for counting comparable domain terms in the benchmark.
    It does not modify your main corpus.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(DASH_TRANSLATION)
    text = text.translate(SUBSCRIPT_SUPERSCRIPT_TRANSLATION)
    return text


def stable_file_id(path: Path) -> str:
    """
    A short deterministic id for filenames in text_samples/.
    """
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()
    return digest[:12]


def collect_pdf_files(input_dirs: Iterable[str | Path]) -> list[Path]:
    """
    Collect PDF files recursively from one or more files/folders.
    """
    paths: list[Path] = []

    for input_path in input_dirs:
        p = Path(input_path)

        if p.is_file() and p.suffix.lower() == ".pdf":
            paths.append(p)
        elif p.is_dir():
            paths.extend(p.rglob("*.pdf"))
            paths.extend(p.rglob("*.PDF"))
        else:
            print(f"Warning: input path does not exist or is not a PDF/folder: {p}")

    return sorted(set(paths), key=lambda x: x.as_posix())


def sample_files(files: list[Path], sample_size: int | None, seed: int) -> list[Path]:
    """
    Deterministically sample files for a small benchmark.
    """
    files = sorted(files, key=lambda x: x.as_posix())

    if sample_size is None or sample_size <= 0 or sample_size >= len(files):
        return files

    rng = random.Random(seed)
    sampled = rng.sample(files, sample_size)
    return sorted(sampled, key=lambda x: x.as_posix())

def extract_with_pypdf(pdf_path: Path) -> ExtractionResult:
    """
    Extract text with pypdf.

    pypdf may emit warnings such as:
    'Impossible to decode XFormObject ...'
    These are captured and written to the CSV instead of flooding the console.
    Page-level extraction errors are logged, but the remaining pages are still processed.
    """
    start = time.perf_counter()

    try:
        from pypdf import PdfReader
        try:
            from pypdf.errors import PdfReadWarning
        except Exception:
            PdfReadWarning = Warning
    except ImportError:
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pypdf",
            status="import_error",
            text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error="pypdf is not installed. Run: pip install pypdf",
        )

    captured_messages: list[str] = []
    page_texts: list[str] = []

    # Capture pypdf logging messages.
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    pypdf_logger = logging.getLogger("pypdf")
    old_level = pypdf_logger.level
    pypdf_logger.addHandler(log_handler)
    pypdf_logger.setLevel(logging.WARNING)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PdfReadWarning)
            warnings.simplefilter("always", UserWarning)

            reader = PdfReader(str(pdf_path), strict=False)

            for page_idx, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:  # keep processing other pages
                    page_text = ""
                    captured_messages.append(
                        f"page {page_idx}: extract_text failed: {repr(exc)}"
                    )

                page_texts.append(page_text)

            for w in caught:
                captured_messages.append(str(w.message))

        log_handler.flush()
        logged = log_stream.getvalue().strip()
        if logged:
            captured_messages.extend(
                line.strip() for line in logged.splitlines() if line.strip()
            )

        text = "\n".join(page_texts)
        elapsed = time.perf_counter() - start

        unique_messages = sorted(set(captured_messages))
        status = "ok_with_warnings" if unique_messages else "ok"

        return ExtractionResult(
            backend="pypdf",
            status=status,
            text=text,
            page_texts=page_texts,
            elapsed_seconds=elapsed,
            error=" | ".join(unique_messages)[:5000],
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pypdf",
            status="error",
            text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
        )

    finally:
        pypdf_logger.removeHandler(log_handler)
        pypdf_logger.setLevel(old_level)


def extract_with_pymupdf(pdf_path: Path) -> ExtractionResult:
    """
    Extract text with PyMuPDF.
    """
    start = time.perf_counter()

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pymupdf",
            status="import_error",
            text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error="PyMuPDF is not installed. Run: pip install pymupdf",
        )

    try:
        page_texts = []

        with fitz.open(pdf_path) as doc:
            for page in doc:
                # sort=True usually improves reading order for multi-column PDFs.
                page_text = extract_page_text_pymupdf_columns(page)
                page_texts.append(page_text)

        text = "\n".join(page_texts)
        elapsed = time.perf_counter() - start

        return ExtractionResult(
            backend="pymupdf",
            status="ok",
            text=text,
            page_texts=page_texts,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pymupdf",
            status="error",
            text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
        )


def count_terms(text: str) -> dict[str, int]:
    """
    Count normalized scientific/domain terms.
    """
    normalized = normalize_scientific_text(text)
    counts = {}

    for term, pattern in TERM_PATTERNS.items():
        counts[term] = len(re.findall(pattern, normalized, flags=re.IGNORECASE))

    return counts


def compute_metrics(
    pdf_path: Path,
    result: ExtractionResult,
    *,
    short_page_chars: int,
    suspicious_total_chars: int,
) -> dict[str, object]:
    """
    Compute extraction quality metrics for one PDF/backend pair.
    """
    text = result.text
    normalized_text = normalize_scientific_text(text)
    page_texts = result.page_texts

    n_pages = len(page_texts)
    page_char_counts = [len(p) for p in page_texts]
    empty_pages = sum(1 for n in page_char_counts if n == 0)
    short_pages = sum(1 for n in page_char_counts if 0 < n < short_page_chars)

    total_chars = len(text)
    normalized_chars = len(normalized_text)
    words = re.findall(r"\b\w+\b", normalized_text)
    total_words = len(words)

    alphabetic_chars = sum(ch.isalpha() for ch in normalized_text)
    digit_chars = sum(ch.isdigit() for ch in normalized_text)
    replacement_chars = normalized_text.count("\ufffd")
    newline_count = text.count("\n")

    alpha_ratio = alphabetic_chars / total_chars if total_chars else 0.0
    digit_ratio = digit_chars / total_chars if total_chars else 0.0
    empty_page_ratio = empty_pages / n_pages if n_pages else 0.0
    short_page_ratio = short_pages / n_pages if n_pages else 0.0

    term_counts = count_terms(text)
    domain_term_hits = sum(term_counts.values())

    suspicious = (
        result.status not in {"ok", "ok_with_warnings"}
        or total_chars < suspicious_total_chars
        or (n_pages > 0 and empty_page_ratio > 0.50)
        or (n_pages > 0 and short_page_ratio > 0.50)
        or alpha_ratio < 0.30
    )

    row: dict[str, object] = {
        "file_path": pdf_path.as_posix(),
        "filename": pdf_path.name,
        "backend": result.backend,
        "status": result.status,
        "error": result.error,
        "elapsed_seconds": round(result.elapsed_seconds, 6),
        "n_pages": n_pages,
        "total_chars": total_chars,
        "normalized_chars": normalized_chars,
        "total_words": total_words,
        "avg_chars_per_page": round(total_chars / n_pages, 2) if n_pages else 0.0,
        "empty_pages": empty_pages,
        "short_pages": short_pages,
        "empty_page_ratio": round(empty_page_ratio, 4),
        "short_page_ratio": round(short_page_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "replacement_chars": replacement_chars,
        "newline_count": newline_count,
        "domain_term_hits": domain_term_hits,
        "suspicious": suspicious,
    }

    for term, count in term_counts.items():
        row[f"term__{term}"] = count

    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """
    Write a list of dictionaries as CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_wide_rows(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Convert long rows into one row per PDF with backend-specific columns.
    """
    by_file: dict[str, dict[str, dict[str, object]]] = {}

    for row in long_rows:
        file_path = str(row["file_path"])
        backend = str(row["backend"])
        by_file.setdefault(file_path, {})[backend] = row

    wide_rows = []

    for file_path, backend_rows in sorted(by_file.items()):
        base = {
            "file_path": file_path,
            "filename": Path(file_path).name,
        }

        for backend in ["pypdf", "pymupdf"]:
            row = backend_rows.get(backend, {})
            for key, value in row.items():
                if key in {"file_path", "filename", "backend"}:
                    continue
                base[f"{backend}__{key}"] = value

        pypdf_chars = int(base.get("pypdf__total_chars") or 0)
        pymupdf_chars = int(base.get("pymupdf__total_chars") or 0)
        pypdf_terms = int(base.get("pypdf__domain_term_hits") or 0)
        pymupdf_terms = int(base.get("pymupdf__domain_term_hits") or 0)

        base["delta_chars_pymupdf_minus_pypdf"] = pymupdf_chars - pypdf_chars
        base["delta_domain_terms_pymupdf_minus_pypdf"] = pymupdf_terms - pypdf_terms

        if pymupdf_chars > pypdf_chars:
            base["more_chars_backend"] = "pymupdf"
        elif pypdf_chars > pymupdf_chars:
            base["more_chars_backend"] = "pypdf"
        else:
            base["more_chars_backend"] = "tie"

        if pymupdf_terms > pypdf_terms:
            base["more_domain_terms_backend"] = "pymupdf"
        elif pypdf_terms > pymupdf_terms:
            base["more_domain_terms_backend"] = "pypdf"
        else:
            base["more_domain_terms_backend"] = "tie"

        wide_rows.append(base)

    return wide_rows


def summarize_by_backend(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Aggregate quality metrics by backend.
    """
    backends = sorted({str(row["backend"]) for row in long_rows})
    summary_rows = []

    for backend in backends:
        rows = [row for row in long_rows if row["backend"] == backend]

        def nums(key: str) -> list[float]:
            values = []
            for row in rows:
                value = row.get(key)
                if value in ("", None):
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
            return values

        total_chars = nums("total_chars")
        total_words = nums("total_words")
        domain_hits = nums("domain_term_hits")
        elapsed = nums("elapsed_seconds")
        short_ratio = nums("short_page_ratio")
        empty_ratio = nums("empty_page_ratio")
        alpha_ratio = nums("alpha_ratio")

        ok_files = sum(1 for row in rows if row.get("status") == "ok")
        suspicious_files = sum(1 for row in rows if str(row.get("suspicious")).lower() == "true")

        summary_rows.append(
            {
                "backend": backend,
                "n_files": len(rows),
                "ok_files": ok_files,
                "error_or_import_error_files": len(rows) - ok_files,
                "suspicious_files": suspicious_files,
                "total_chars_sum": int(sum(total_chars)),
                "mean_chars_per_file": round(statistics.mean(total_chars), 2) if total_chars else 0,
                "median_chars_per_file": round(statistics.median(total_chars), 2) if total_chars else 0,
                "mean_words_per_file": round(statistics.mean(total_words), 2) if total_words else 0,
                "mean_domain_term_hits": round(statistics.mean(domain_hits), 2) if domain_hits else 0,
                "total_elapsed_seconds": round(sum(elapsed), 2),
                "mean_elapsed_seconds": round(statistics.mean(elapsed), 4) if elapsed else 0,
                "mean_short_page_ratio": round(statistics.mean(short_ratio), 4) if short_ratio else 0,
                "mean_empty_page_ratio": round(statistics.mean(empty_ratio), 4) if empty_ratio else 0,
                "mean_alpha_ratio": round(statistics.mean(alpha_ratio), 4) if alpha_ratio else 0,
            }
        )

    return summary_rows


def summarize_term_counts(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Aggregate individual term counts by backend.
    """
    rows_out = []
    backends = sorted({str(row["backend"]) for row in long_rows})

    for backend in backends:
        backend_rows = [row for row in long_rows if row["backend"] == backend]

        for term in TERM_PATTERNS:
            col = f"term__{term}"
            values = []
            for row in backend_rows:
                try:
                    values.append(int(row.get(col, 0)))
                except (TypeError, ValueError):
                    values.append(0)

            rows_out.append(
                {
                    "backend": backend,
                    "term": term,
                    "total_count": sum(values),
                    "files_with_term": sum(1 for v in values if v > 0),
                    "mean_count_per_file": round(statistics.mean(values), 4) if values else 0,
                }
            )

    return rows_out


def save_text_samples(
    pdf_path: Path,
    results: list[ExtractionResult],
    sample_dir: Path,
) -> None:
    """
    Save raw extracted text for manual side-by-side inspection.
    """
    file_id = stable_file_id(pdf_path)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem)[:80]

    for result in results:
        filename = f"{safe_stem}__{file_id}__{result.backend}.txt"
        out_path = sample_dir / filename

        header = (
            f"PDF: {pdf_path.as_posix()}\n"
            f"Backend: {result.backend}\n"
            f"Status: {result.status}\n"
            f"Error: {result.error}\n"
            f"{'=' * 80}\n\n"
        )

        out_path.write_text(header + result.text, encoding="utf-8", errors="replace")


def run_benchmark(args: argparse.Namespace) -> None:
    input_files = collect_pdf_files(args.input_dirs)
    selected_files = sample_files(input_files, args.sample_size, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found PDF files: {len(input_files)}")
    print(f"Benchmark PDF files: {len(selected_files)}")
    print(f"Output directory: {out_dir}")

    long_rows: list[dict[str, object]] = []
    sample_dir = out_dir / "text_samples"

    if args.save_text_samples > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    for i, pdf_path in enumerate(selected_files, start=1):
        if i == 1 or i % args.progress_every == 0:
            print(f"[{i}/{len(selected_files)}] {pdf_path}")

        results = [
            extract_with_pypdf(pdf_path),
            extract_with_pymupdf(pdf_path),
        ]

        for result in results:
            row = compute_metrics(
                pdf_path,
                result,
                short_page_chars=args.short_page_chars,
                suspicious_total_chars=args.suspicious_total_chars,
            )
            long_rows.append(row)

        if args.save_text_samples > 0 and i <= args.save_text_samples:
            save_text_samples(pdf_path, results, sample_dir)

    wide_rows = make_wide_rows(long_rows)
    summary_rows = summarize_by_backend(long_rows)
    term_rows = summarize_term_counts(long_rows)

    write_csv(long_rows, out_dir / "extraction_metrics_long.csv")
    write_csv(wide_rows, out_dir / "extraction_metrics_wide.csv")
    write_csv(summary_rows, out_dir / "summary_by_backend.csv")
    write_csv(term_rows, out_dir / "term_counts_by_backend.csv")

    print("\nDone.")
    print(f"Wrote: {out_dir / 'extraction_metrics_long.csv'}")
    print(f"Wrote: {out_dir / 'extraction_metrics_wide.csv'}")
    print(f"Wrote: {out_dir / 'summary_by_backend.csv'}")
    print(f"Wrote: {out_dir / 'term_counts_by_backend.csv'}")

    if args.save_text_samples > 0:
        print(f"Wrote text samples to: {sample_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pypdf and PyMuPDF extraction quality on scientific PDFs."
    )

    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more PDF files/folders. Folders are searched recursively.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/pdf_extraction_benchmark",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help=(
            "Number of PDFs to benchmark. Use 0 or a value >= corpus size "
            "to process all PDFs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic file sampling.",
    )
    parser.add_argument(
        "--short-page-chars",
        type=int,
        default=300,
        help="A page with fewer extracted chars than this is counted as short.",
    )
    parser.add_argument(
        "--suspicious-total-chars",
        type=int,
        default=1000,
        help="A PDF with fewer extracted chars than this is suspicious.",
    )
    parser.add_argument(
        "--save-text-samples",
        type=int,
        default=10,
        help="Save raw extracted text for the first N sampled PDFs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N PDFs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
