#!/usr/bin/env python3
"""
Compare pypdf and PyMuPDF extraction quality before and after text cleanup.

This script is independent from the main Word2Vec pipeline. It does not modify
your corpus. It extracts text with multiple backends and computes both:

1. raw metrics:
   metrics directly on extracted text;

2. cleaned metrics:
   metrics after scientific-text cleanup:
     - Unicode NFKC normalization;
     - dash normalization;
     - subscript/superscript normalization;
     - conservative PDF hyphenation repair;
     - inline numeric citation removal;
     - whitespace normalization.

This is useful when one backend extracts more raw words because it separates
citation markers such as "70,71)" into standalone tokens. After cleanup, those
tokens should no longer inflate word counts.

Install dependencies:
    pip install pypdf pymupdf

Example:
    python compare_pdf_extractors_after_cleanup.py \
      --input-dirs data/dental_pre2018 data/dental_post2018 \
      --sample-size 200 \
      --seed 42 \
      --out-dir outputs/pdf_extraction_benchmark_cleanup \
      --save-text-samples 10

Outputs:
    extraction_metrics_long.csv
        One row per PDF per backend. Contains raw_* and clean_* metrics.

    extraction_metrics_wide.csv
        One row per PDF with backend-specific columns.

    summary_by_backend.csv
        Aggregate raw and cleaned statistics per backend.

    term_counts_by_backend.csv
        Aggregate raw and cleaned domain-term counts per backend.

    backend_pair_comparison_after_cleanup.csv
        One row per PDF comparing pypdf vs pymupdf after cleanup.

    top_extra_tokens_after_cleanup.csv
        For each PDF, most frequent tokens that one backend has more than
        the other after cleanup.

    text_samples/
        Optional raw and cleaned text samples for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import random
import re
import statistics
import time
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any


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
    "tricalcium_silicate": r"\btricalcium\s+silicate\b",
    "calcium_hydroxide": r"\bcalcium\s+hydroxide\b",
    "resin_composite": r"\bresin\s+composite(?:s)?\b",
    "short_fiber": r"\bshort\s*-\s*fiber\b|\bshort\s+fiber\b",
}


DOMAIN_TOKEN_SUBSTRINGS = [
    "zirconia",
    "yttria",
    "translucency",
    "flexural",
    "strength",
    "fracture",
    "toughness",
    "resin",
    "composite",
    "polymerization",
    "fiber",
    "tricalcium",
    "silicate",
    "calcium",
    "hydroxide",
    "bismuth",
    "oxide",
    "radiopacity",
    "biocompatibility",
    "3y-tzp",
    "4y-tzp",
    "5y-tzp",
    "3y-psz",
    "4y-psz",
    "5y-psz",
    "y-tzp",
    "y-psz",
    "h2o",
    "caoh2",
]


@dataclass
class ExtractionResult:
    backend: str
    status: str
    text: str
    page_texts: list[str]
    elapsed_seconds: float
    error: str = ""


def normalize_scientific_unicode(text: str) -> str:
    """
    Normalize Unicode variants relevant for scientific PDF extraction.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(DASH_TRANSLATION)
    text = text.translate(SUBSCRIPT_SUPERSCRIPT_TRANSLATION)
    return text


def count_inline_citation_markers(text: str) -> int:
    """
    Count common numeric citation markers before removing them.
    This is approximate and intended for diagnostics only.
    """
    patterns = [
        # Word immediately followed by citations: apicoectomy70,71)
        r"(?<=[A-Za-z])\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
        # Bracketed references: [70], [70,71], [75-78]
        r"\[\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\]",
        # Parenthesized references: (70), (70, 71), (75-78)
        r"\(\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
    ]

    return sum(len(re.findall(p, text)) for p in patterns)


def remove_inline_citations(text: str) -> str:
    """
    Remove common numeric citation markers from scientific PDF text.

    Handles examples:
    - apicoectomy70,71) -> apicoectomy
    - apicoectomy 70,71) -> apicoectomy
    - pulp capping72) -> pulp capping
    - biocompatibility75-78) -> biocompatibility
    - [70,71] -> removed
    - (70, 71) -> removed

    This deliberately targets short numeric groups typical of references.
    It does not remove chemical/materials tokens such as 3Y-TZP or H2O.
    """
    # Word immediately followed by one or more citation numbers and a closing parenthesis.
    text = re.sub(
        r"(?<=[A-Za-z])\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
        "",
        text,
    )

    # Bracketed citation lists.
    text = re.sub(
        r"\[\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\]",
        " ",
        text,
    )

    # Parenthesized citation lists.
    text = re.sub(
        r"\(\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
        " ",
        text,
    )

    return text


def fix_pdf_hyphenation_conservative(text: str) -> str:
    """
    Conservatively repair line-break artifacts.

    - Ordinary lowercase word split:
        infor-\\nmation -> information

    - Meaningful scientific/materials hyphens are preserved:
        3Y-\\nTZP -> 3Y-TZP
        short-\\nfiber -> shortfiber?  (not ideal)
      For this reason we only remove hyphen if both sides are lowercase
      and the left/right fragments look like ordinary word fragments.

    Note: This function is intentionally conservative for scientific terms.
    """
    # Remove hyphen only in likely ordinary lowercase hyphenation.
    # Require at least two lowercase letters before and after the break
    # to avoid damaging "Y-\nTZP" or "3Y-\nTZP".
    text = re.sub(r"(?<=[a-z]{2})-\s*\n\s*(?=[a-z]{2})", "", text)

    # Preserve all other hyphenated line breaks.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)

    # Remaining line breaks are spaces.
    text = re.sub(r"\s*\n\s*", " ", text)

    return text


def cleanup_extracted_text(text: str) -> str:
    """
    Cleanup used for comparing extraction backends.

    This should approximate the text that will later be tokenized for Word2Vec,
    without using spaCy. It removes citation markers that can inflate word counts
    in pypdf.
    """
    text = normalize_scientific_unicode(text)
    text = fix_pdf_hyphenation_conservative(text)
    text = remove_inline_citations(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokens(text: str) -> list[str]:
    """
    Lightweight tokenization for benchmark metrics.

    Keeps hyphenated scientific terms such as 3Y-TZP as one token when possible.
    """
    text = normalize_scientific_unicode(text).lower()
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)


def token_counter(text: str) -> Counter:
    return Counter(simple_tokens(text))


def garbage_like_token(token: str) -> bool:
    """
    Heuristic diagnostic for tokens likely to be unhelpful for Word2Vec.
    """
    if len(token) <= 1:
        return True

    if token.isdigit():
        return True

    if len(token) > 35:
        return True

    digit_ratio = sum(ch.isdigit() for ch in token) / len(token)
    if digit_ratio > 0.5:
        return True

    # Long consonant-like fragments often come from broken encodings/URLs/IDs.
    if len(token) >= 10 and not re.search(r"[aeiouy]", token):
        return True

    # DOI/URL-like fragments.
    if token.startswith(("http", "www", "doi", "pmid", "pmc")):
        return True

    return False


def domain_like_token(token: str) -> bool:
    token = token.lower()
    compact = token.replace("-", "").replace("_", "")
    return any(s in token or s.replace("-", "") in compact for s in DOMAIN_TOKEN_SUBSTRINGS)


def count_domain_tokens(counter: Counter) -> int:
    return sum(count for token, count in counter.items() if domain_like_token(token))


def stable_file_id(path: Path) -> str:
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()
    return digest[:12]


def collect_pdf_files(input_dirs: Iterable[str | Path]) -> list[Path]:
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
    files = sorted(files, key=lambda x: x.as_posix())

    if sample_size is None or sample_size <= 0 or sample_size >= len(files):
        return files

    rng = random.Random(seed)
    sampled = rng.sample(files, sample_size)
    return sorted(sampled, key=lambda x: x.as_posix())


def extract_with_pypdf(pdf_path: Path) -> ExtractionResult:
    """
    Extract text with pypdf and capture warnings/log messages.
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


def extract_page_text_pymupdf_columns(page: Any) -> str:
    """
    Extract text from a PyMuPDF page using a simple column-aware reading order.

    Reading order:
      - full-width blocks top-to-bottom;
      - between full-width blocks:
          left column top-to-bottom, then right column top-to-bottom.
    """
    page_rect = page.rect
    page_width = page_rect.width
    mid_x = (page_rect.x0 + page_rect.x1) / 2

    raw_blocks = page.get_text("blocks", sort=False)

    blocks = []
    for block in raw_blocks:
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

    full_width_blocks = [
        b for b in blocks
        if b["width"] >= 0.65 * page_width
    ]

    column_blocks = [
        b for b in blocks
        if b["width"] < 0.65 * page_width
    ]

    full_width_blocks = sorted(full_width_blocks, key=lambda b: (b["y0"], b["x0"]))

    def emit_column_region(region_blocks: list[dict]) -> list[str]:
        if not region_blocks:
            return []

        left = [b for b in region_blocks if b["x_center"] < mid_x]
        right = [b for b in region_blocks if b["x_center"] >= mid_x]

        left = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right = sorted(right, key=lambda b: (b["y0"], b["x0"]))

        return [b["text"] for b in left + right]

    output_parts = []
    last_y = page_rect.y0

    for full_block in full_width_blocks:
        region = [
            b for b in column_blocks
            if last_y <= b["y0"] < full_block["y0"]
        ]
        output_parts.extend(emit_column_region(region))
        output_parts.append(full_block["text"])
        last_y = max(last_y, full_block["y1"])

    region = [b for b in column_blocks if b["y0"] >= last_y]
    output_parts.extend(emit_column_region(region))

    return "\n\n".join(output_parts)


def extract_with_pymupdf_columns(pdf_path: Path) -> ExtractionResult:
    """
    Extract text with PyMuPDF using column-aware block ordering.
    """
    start = time.perf_counter()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pymupdf_columns",
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
                page_texts.append(extract_page_text_pymupdf_columns(page))

        text = "\n".join(page_texts)
        elapsed = time.perf_counter() - start

        return ExtractionResult(
            backend="pymupdf_columns",
            status="ok",
            text=text,
            page_texts=page_texts,
            elapsed_seconds=elapsed,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return ExtractionResult(
            backend="pymupdf_columns",
            status="error",
            text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
        )


def count_terms(text: str) -> dict[str, int]:
    normalized = normalize_scientific_unicode(text)
    counts = {}

    for term, pattern in TERM_PATTERNS.items():
        counts[term] = len(re.findall(pattern, normalized, flags=re.IGNORECASE))

    return counts


def metric_block(prefix: str, text: str, page_texts: list[str] | None = None) -> dict[str, object]:
    """
    Compute a consistent metric block for raw or cleaned text.
    """
    tokens = simple_tokens(text)
    counter = Counter(tokens)

    total_chars = len(text)
    total_words = len(tokens)
    alphabetic_chars = sum(ch.isalpha() for ch in text)
    digit_chars = sum(ch.isdigit() for ch in text)
    replacement_chars = text.count("\ufffd")
    newline_count = text.count("\n")
    numeric_tokens = sum(1 for t in tokens if t.isdigit())
    garbage_tokens = sum(count for token, count in counter.items() if garbage_like_token(token))
    domain_tokens = count_domain_tokens(counter)
    term_counts = count_terms(text)
    domain_term_hits = sum(term_counts.values())

    out: dict[str, object] = {
        f"{prefix}_total_chars": total_chars,
        f"{prefix}_total_words": total_words,
        f"{prefix}_unique_tokens": len(counter),
        f"{prefix}_alpha_ratio": round(alphabetic_chars / total_chars, 4) if total_chars else 0.0,
        f"{prefix}_digit_ratio": round(digit_chars / total_chars, 4) if total_chars else 0.0,
        f"{prefix}_replacement_chars": replacement_chars,
        f"{prefix}_newline_count": newline_count,
        f"{prefix}_numeric_tokens": numeric_tokens,
        f"{prefix}_garbage_like_tokens": garbage_tokens,
        f"{prefix}_garbage_like_token_ratio": round(garbage_tokens / total_words, 4) if total_words else 0.0,
        f"{prefix}_domain_like_tokens": domain_tokens,
        f"{prefix}_domain_term_hits": domain_term_hits,
    }

    if page_texts is not None:
        page_char_counts = [len(p) for p in page_texts]
        n_pages = len(page_texts)
        empty_pages = sum(1 for n in page_char_counts if n == 0)

        out.update(
            {
                f"{prefix}_n_pages": n_pages,
                f"{prefix}_avg_chars_per_page": round(total_chars / n_pages, 2) if n_pages else 0.0,
                f"{prefix}_empty_pages": empty_pages,
                f"{prefix}_empty_page_ratio": round(empty_pages / n_pages, 4) if n_pages else 0.0,
            }
        )

    for term, count in term_counts.items():
        safe_term = term.replace(" ", "_")
        out[f"{prefix}_term__{safe_term}"] = count

    return out


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
    raw_text = result.text
    clean_text = cleanup_extracted_text(raw_text)

    page_texts = result.page_texts
    page_char_counts = [len(p) for p in page_texts]
    n_pages = len(page_texts)
    short_pages = sum(1 for n in page_char_counts if 0 < n < short_page_chars)
    short_page_ratio = short_pages / n_pages if n_pages else 0.0

    row: dict[str, object] = {
        "file_path": pdf_path.as_posix(),
        "filename": pdf_path.name,
        "backend": result.backend,
        "status": result.status,
        "error": result.error,
        "elapsed_seconds": round(result.elapsed_seconds, 6),
        "citation_markers_before_cleanup": count_inline_citation_markers(raw_text),
        "clean_chars_removed": max(0, len(raw_text) - len(clean_text)),
        "short_pages": short_pages,
        "short_page_ratio": round(short_page_ratio, 4),
    }

    row.update(metric_block("raw", raw_text, page_texts=page_texts))
    row.update(metric_block("clean", clean_text, page_texts=None))

    clean_total_chars = int(row["clean_total_chars"])
    clean_alpha_ratio = float(row["clean_alpha_ratio"])

    suspicious = (
        result.status not in {"ok", "ok_with_warnings"}
        or clean_total_chars < suspicious_total_chars
        or (n_pages > 0 and short_page_ratio > 0.50)
        or clean_alpha_ratio < 0.30
    )

    row["suspicious"] = suspicious

    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
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
    by_file: dict[str, dict[str, dict[str, object]]] = {}

    for row in long_rows:
        file_path = str(row["file_path"])
        backend = str(row["backend"])
        by_file.setdefault(file_path, {})[backend] = row

    backends = sorted({str(row["backend"]) for row in long_rows})
    wide_rows = []

    for file_path, backend_rows in sorted(by_file.items()):
        base = {
            "file_path": file_path,
            "filename": Path(file_path).name,
        }

        for backend in backends:
            row = backend_rows.get(backend, {})
            for key, value in row.items():
                if key in {"file_path", "filename", "backend"}:
                    continue
                base[f"{backend}__{key}"] = value

        if "pypdf" in backend_rows and "pymupdf_columns" in backend_rows:
            pypdf_clean_words = int(base.get("pypdf__clean_total_words") or 0)
            pymu_clean_words = int(base.get("pymupdf_columns__clean_total_words") or 0)

            pypdf_clean_terms = int(base.get("pypdf__clean_domain_term_hits") or 0)
            pymu_clean_terms = int(base.get("pymupdf_columns__clean_domain_term_hits") or 0)

            base["delta_clean_words_pymupdf_minus_pypdf"] = pymu_clean_words - pypdf_clean_words
            base["delta_clean_domain_terms_pymupdf_minus_pypdf"] = pymu_clean_terms - pypdf_clean_terms

        wide_rows.append(base)

    return wide_rows


def summarize_by_backend(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
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

        ok_files = sum(1 for row in rows if row.get("status") in {"ok", "ok_with_warnings"})
        warning_files = sum(1 for row in rows if row.get("status") == "ok_with_warnings")
        suspicious_files = sum(1 for row in rows if str(row.get("suspicious")).lower() == "true")

        summary: dict[str, object] = {
            "backend": backend,
            "n_files": len(rows),
            "ok_or_warning_files": ok_files,
            "warning_files": warning_files,
            "error_or_import_error_files": len(rows) - ok_files,
            "suspicious_files": suspicious_files,
            "total_elapsed_seconds": round(sum(nums("elapsed_seconds")), 2),
            "mean_elapsed_seconds": round(statistics.mean(nums("elapsed_seconds")), 4) if nums("elapsed_seconds") else 0,
        }

        for prefix in ["raw", "clean"]:
            for key in [
                "total_chars",
                "total_words",
                "unique_tokens",
                "domain_term_hits",
                "domain_like_tokens",
                "numeric_tokens",
                "garbage_like_tokens",
                "garbage_like_token_ratio",
                "alpha_ratio",
                "digit_ratio",
            ]:
                values = nums(f"{prefix}_{key}")
                if not values:
                    continue

                summary[f"{prefix}_{key}_sum"] = round(sum(values), 4)
                summary[f"{prefix}_{key}_mean"] = round(statistics.mean(values), 4)
                summary[f"{prefix}_{key}_median"] = round(statistics.median(values), 4)

        citation_values = nums("citation_markers_before_cleanup")
        summary["citation_markers_before_cleanup_sum"] = int(sum(citation_values)) if citation_values else 0
        summary["citation_markers_before_cleanup_mean"] = round(statistics.mean(citation_values), 4) if citation_values else 0

        summary_rows.append(summary)

    return summary_rows


def summarize_term_counts(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_out = []
    backends = sorted({str(row["backend"]) for row in long_rows})

    for backend in backends:
        backend_rows = [row for row in long_rows if row["backend"] == backend]

        for prefix in ["raw", "clean"]:
            for term in TERM_PATTERNS:
                safe_term = term.replace(" ", "_")
                col = f"{prefix}_term__{safe_term}"
                values = []
                for row in backend_rows:
                    try:
                        values.append(int(row.get(col, 0)))
                    except (TypeError, ValueError):
                        values.append(0)

                rows_out.append(
                    {
                        "backend": backend,
                        "metric_type": prefix,
                        "term": term,
                        "total_count": sum(values),
                        "files_with_term": sum(1 for v in values if v > 0),
                        "mean_count_per_file": round(statistics.mean(values), 4) if values else 0,
                    }
                )

    return rows_out


def compare_backend_pair_after_cleanup(
    pdf_path: Path,
    left: ExtractionResult,
    right: ExtractionResult,
    *,
    left_name: str,
    right_name: str,
    top_n: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Compare two backend outputs after cleanup using token counters.
    """
    left_clean = cleanup_extracted_text(left.text)
    right_clean = cleanup_extracted_text(right.text)

    left_counter = token_counter(left_clean)
    right_counter = token_counter(right_clean)

    left_extra = left_counter - right_counter
    right_extra = right_counter - left_counter

    left_extra_total = sum(left_extra.values())
    right_extra_total = sum(right_extra.values())

    left_extra_domain = count_domain_tokens(left_extra)
    right_extra_domain = count_domain_tokens(right_extra)

    left_extra_garbage = sum(count for token, count in left_extra.items() if garbage_like_token(token))
    right_extra_garbage = sum(count for token, count in right_extra.items() if garbage_like_token(token))

    comparison_row = {
        "file_path": pdf_path.as_posix(),
        "filename": pdf_path.name,
        "left_backend": left_name,
        "right_backend": right_name,
        f"{left_name}_clean_tokens": sum(left_counter.values()),
        f"{right_name}_clean_tokens": sum(right_counter.values()),
        f"{left_name}_extra_clean_tokens": left_extra_total,
        f"{right_name}_extra_clean_tokens": right_extra_total,
        f"{left_name}_extra_domain_like_tokens": left_extra_domain,
        f"{right_name}_extra_domain_like_tokens": right_extra_domain,
        f"{left_name}_extra_garbage_like_tokens": left_extra_garbage,
        f"{right_name}_extra_garbage_like_tokens": right_extra_garbage,
        f"{left_name}_extra_garbage_ratio": round(left_extra_garbage / left_extra_total, 4) if left_extra_total else 0.0,
        f"{right_name}_extra_garbage_ratio": round(right_extra_garbage / right_extra_total, 4) if right_extra_total else 0.0,
    }

    top_rows = []

    for backend, extra in [(left_name, left_extra), (right_name, right_extra)]:
        for rank, (token, count) in enumerate(extra.most_common(top_n), start=1):
            top_rows.append(
                {
                    "file_path": pdf_path.as_posix(),
                    "filename": pdf_path.name,
                    "backend_with_extra_token": backend,
                    "rank": rank,
                    "token": token,
                    "count": count,
                    "domain_like": domain_like_token(token),
                    "garbage_like": garbage_like_token(token),
                }
            )

    return comparison_row, top_rows


def save_text_samples(
    pdf_path: Path,
    results: list[ExtractionResult],
    sample_dir: Path,
) -> None:
    file_id = stable_file_id(pdf_path)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem)[:80]

    for result in results:
        for version, text in [
            ("raw", result.text),
            ("clean", cleanup_extracted_text(result.text)),
        ]:
            filename = f"{safe_stem}__{file_id}__{result.backend}__{version}.txt"
            out_path = sample_dir / filename

            header = (
                f"PDF: {pdf_path.as_posix()}\n"
                f"Backend: {result.backend}\n"
                f"Version: {version}\n"
                f"Status: {result.status}\n"
                f"Error: {result.error}\n"
                f"{'=' * 80}\n\n"
            )

            out_path.write_text(header + text, encoding="utf-8", errors="replace")


def run_benchmark(args: argparse.Namespace) -> None:
    input_files = collect_pdf_files(args.input_dirs)
    selected_files = sample_files(input_files, args.sample_size, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found PDF files: {len(input_files)}")
    print(f"Benchmark PDF files: {len(selected_files)}")
    print(f"Output directory: {out_dir}")

    long_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    top_extra_rows: list[dict[str, object]] = []

    sample_dir = out_dir / "text_samples"
    if args.save_text_samples > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    for i, pdf_path in enumerate(selected_files, start=1):
        if i == 1 or i % args.progress_every == 0:
            print(f"[{i}/{len(selected_files)}] {pdf_path}")

        pypdf_result = extract_with_pypdf(pdf_path)
        pymupdf_result = extract_with_pymupdf_columns(pdf_path)
        results = [pypdf_result, pymupdf_result]

        for result in results:
            row = compute_metrics(
                pdf_path,
                result,
                short_page_chars=args.short_page_chars,
                suspicious_total_chars=args.suspicious_total_chars,
            )
            long_rows.append(row)

        pair_row, token_rows = compare_backend_pair_after_cleanup(
            pdf_path,
            pypdf_result,
            pymupdf_result,
            left_name="pypdf",
            right_name="pymupdf_columns",
            top_n=args.top_extra_tokens,
        )
        pair_rows.append(pair_row)
        top_extra_rows.extend(token_rows)

        if args.save_text_samples > 0 and i <= args.save_text_samples:
            save_text_samples(pdf_path, results, sample_dir)

    wide_rows = make_wide_rows(long_rows)
    summary_rows = summarize_by_backend(long_rows)
    term_rows = summarize_term_counts(long_rows)

    write_csv(long_rows, out_dir / "extraction_metrics_long.csv")
    write_csv(wide_rows, out_dir / "extraction_metrics_wide.csv")
    write_csv(summary_rows, out_dir / "summary_by_backend.csv")
    write_csv(term_rows, out_dir / "term_counts_by_backend.csv")
    write_csv(pair_rows, out_dir / "backend_pair_comparison_after_cleanup.csv")
    write_csv(top_extra_rows, out_dir / "top_extra_tokens_after_cleanup.csv")

    print("\nDone.")
    print(f"Wrote: {out_dir / 'extraction_metrics_long.csv'}")
    print(f"Wrote: {out_dir / 'extraction_metrics_wide.csv'}")
    print(f"Wrote: {out_dir / 'summary_by_backend.csv'}")
    print(f"Wrote: {out_dir / 'term_counts_by_backend.csv'}")
    print(f"Wrote: {out_dir / 'backend_pair_comparison_after_cleanup.csv'}")
    print(f"Wrote: {out_dir / 'top_extra_tokens_after_cleanup.csv'}")

    if args.save_text_samples > 0:
        print(f"Wrote text samples to: {sample_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pypdf and column-aware PyMuPDF before and after cleanup."
    )

    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more PDF files/folders. Folders are searched recursively.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/pdf_extraction_benchmark_cleanup",
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
        help="A PDF with fewer cleaned chars than this is suspicious.",
    )
    parser.add_argument(
        "--save-text-samples",
        type=int,
        default=10,
        help="Save raw and cleaned text for the first N sampled PDFs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N PDFs.",
    )
    parser.add_argument(
        "--top-extra-tokens",
        type=int,
        default=30,
        help="Number of backend-specific extra tokens to save per PDF/backend.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
