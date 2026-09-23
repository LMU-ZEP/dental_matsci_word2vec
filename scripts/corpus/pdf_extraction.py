"""
PDF text extraction backends for the Word2Vec article pipeline.

For the canonical manuscript pipeline, the preferred backend is
pymupdf_columns, with pypdf used as a fallback.
"""

from __future__ import annotations

import io
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_normalization import cleanup_extracted_text, count_inline_citation_markers


@dataclass
class PDFExtractionResult:
    requested_backend: str
    backend_used: str
    status: str
    raw_text: str
    cleaned_text: str
    page_texts: list[str]
    elapsed_seconds: float
    error: str = ""
    fallback_used: bool = False
    citation_markers_before_cleanup: int = 0

    @property
    def n_pages(self) -> int:
        return len(self.page_texts)

    @property
    def raw_chars(self) -> int:
        return len(self.raw_text)

    @property
    def clean_chars(self) -> int:
        return len(self.cleaned_text)


def _finalize_result(
    *,
    requested_backend: str,
    backend_used: str,
    status: str,
    raw_text: str,
    page_texts: list[str],
    elapsed_seconds: float,
    error: str = "",
    fallback_used: bool = False,
    apply_cleanup: bool = True,
) -> PDFExtractionResult:
    cleaned_text = cleanup_extracted_text(raw_text) if apply_cleanup else raw_text
    return PDFExtractionResult(
        requested_backend=requested_backend,
        backend_used=backend_used,
        status=status,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        page_texts=page_texts,
        elapsed_seconds=elapsed_seconds,
        error=error,
        fallback_used=fallback_used,
        citation_markers_before_cleanup=count_inline_citation_markers(raw_text),
    )


def is_suspicious_extraction(
    result: PDFExtractionResult,
    *,
    min_clean_chars: int = 1000,
    min_alpha_ratio: float = 0.30,
) -> bool:
    """Heuristic for deciding whether to try a fallback backend."""
    if result.status not in {"ok", "ok_with_warnings"}:
        return True
    if result.clean_chars < min_clean_chars:
        return True
    if result.clean_chars == 0:
        return True

    alpha_chars = sum(ch.isalpha() for ch in result.cleaned_text)
    alpha_ratio = alpha_chars / result.clean_chars if result.clean_chars else 0.0
    if alpha_ratio < min_alpha_ratio:
        return True

    return False


def _import_pypdf_reader() -> tuple[Any, Any]:
    """
    Prefer pypdf, but fall back to PyPDF2 if an older environment still uses it.
    """
    try:
        from pypdf import PdfReader
        try:
            from pypdf.errors import PdfReadWarning
        except Exception:
            PdfReadWarning = Warning
        return PdfReader, PdfReadWarning
    except ImportError:
        from PyPDF2 import PdfReader
        try:
            from PyPDF2.errors import PdfReadWarning
        except Exception:
            PdfReadWarning = Warning
        return PdfReader, PdfReadWarning


def extract_text_pypdf(
    pdf_path: str | Path,
    *,
    requested_backend: str = "pypdf",
    apply_cleanup: bool = True,
) -> PDFExtractionResult:
    """
    Extract text with pypdf/PyPDF2 and capture warnings/log messages.
    """
    pdf_path = Path(pdf_path)
    start = time.perf_counter()

    try:
        PdfReader, PdfReadWarning = _import_pypdf_reader()
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return _finalize_result(
            requested_backend=requested_backend,
            backend_used="pypdf",
            status="import_error",
            raw_text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
            apply_cleanup=apply_cleanup,
        )

    captured_messages: list[str] = []
    page_texts: list[str] = []

    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    pypdf_logger = logging.getLogger("pypdf")
    pypdf2_logger = logging.getLogger("PyPDF2")
    old_levels = {pypdf_logger: pypdf_logger.level, pypdf2_logger: pypdf2_logger.level}

    for logger in old_levels:
        logger.addHandler(log_handler)
        logger.setLevel(logging.WARNING)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PdfReadWarning)
            warnings.simplefilter("always", UserWarning)

            reader = PdfReader(str(pdf_path), strict=False)
            for page_idx, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:  # keep processing remaining pages
                    page_text = ""
                    captured_messages.append(
                        f"page {page_idx}: extract_text failed: {repr(exc)}"
                    )
                page_texts.append(page_text)

            for warning_item in caught:
                captured_messages.append(str(warning_item.message))

        log_handler.flush()
        logged = log_stream.getvalue().strip()
        if logged:
            captured_messages.extend(
                line.strip() for line in logged.splitlines() if line.strip()
            )

        raw_text = "\n".join(page_texts)
        elapsed = time.perf_counter() - start
        unique_messages = sorted(set(captured_messages))
        status = "ok_with_warnings" if unique_messages else "ok"

        return _finalize_result(
            requested_backend=requested_backend,
            backend_used="pypdf",
            status=status,
            raw_text=raw_text,
            page_texts=page_texts,
            elapsed_seconds=elapsed,
            error=" | ".join(unique_messages)[:5000],
            apply_cleanup=apply_cleanup,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return _finalize_result(
            requested_backend=requested_backend,
            backend_used="pypdf",
            status="error",
            raw_text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
            apply_cleanup=apply_cleanup,
        )

    finally:
        for logger, level in old_levels.items():
            logger.removeHandler(log_handler)
            logger.setLevel(level)


def extract_page_text_pymupdf_columns(page: Any) -> str:
    """
    Extract text from a PyMuPDF page using a simple column-aware reading order.

    This is designed for common scientific PDFs with one- or two-column layouts:
    full-width blocks are emitted top-to-bottom; text blocks between them are
    emitted left column top-to-bottom, then right column top-to-bottom.
    """
    page_rect = page.rect
    page_width = page_rect.width
    mid_x = (page_rect.x0 + page_rect.x1) / 2

    raw_blocks = page.get_text("blocks", sort=False)
    blocks: list[dict[str, Any]] = []

    for block in raw_blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0:  # 0 = text, 1 = image
            continue
        text = (text or "").strip()
        if not text:
            continue
        width = x1 - x0
        blocks.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": width,
                "x_center": (x0 + x1) / 2,
                "text": text,
            }
        )

    if not blocks:
        return ""

    full_width_blocks = [b for b in blocks if b["width"] >= 0.65 * page_width]
    column_blocks = [b for b in blocks if b["width"] < 0.65 * page_width]
    full_width_blocks = sorted(full_width_blocks, key=lambda b: (b["y0"], b["x0"]))

    def emit_column_region(region_blocks: list[dict[str, Any]]) -> list[str]:
        left = [b for b in region_blocks if b["x_center"] < mid_x]
        right = [b for b in region_blocks if b["x_center"] >= mid_x]
        left = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right = sorted(right, key=lambda b: (b["y0"], b["x0"]))
        return [b["text"] for b in left + right]

    output_parts: list[str] = []
    last_y = page_rect.y0

    for full_block in full_width_blocks:
        region = [b for b in column_blocks if last_y <= b["y0"] < full_block["y0"]]
        output_parts.extend(emit_column_region(region))
        output_parts.append(full_block["text"])
        last_y = max(last_y, full_block["y1"])

    region = [b for b in column_blocks if b["y0"] >= last_y]
    output_parts.extend(emit_column_region(region))

    return "\n\n".join(output_parts)


def extract_text_pymupdf(
    pdf_path: str | Path,
    *,
    requested_backend: str = "pymupdf",
    columns: bool = False,
    apply_cleanup: bool = True,
) -> PDFExtractionResult:
    """Extract text with PyMuPDF."""
    pdf_path = Path(pdf_path)
    backend_used = "pymupdf_columns" if columns else "pymupdf"
    start = time.perf_counter()

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return _finalize_result(
            requested_backend=requested_backend,
            backend_used=backend_used,
            status="import_error",
            raw_text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
            apply_cleanup=apply_cleanup,
        )

    try:
        page_texts: list[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if columns:
                    page_text = extract_page_text_pymupdf_columns(page)
                else:
                    page_text = page.get_text("text", sort=True) or ""
                page_texts.append(page_text)

        raw_text = "\n".join(page_texts)
        elapsed = time.perf_counter() - start
        return _finalize_result(
            requested_backend=requested_backend,
            backend_used=backend_used,
            status="ok",
            raw_text=raw_text,
            page_texts=page_texts,
            elapsed_seconds=elapsed,
            apply_cleanup=apply_cleanup,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return _finalize_result(
            requested_backend=requested_backend,
            backend_used=backend_used,
            status="error",
            raw_text="",
            page_texts=[],
            elapsed_seconds=elapsed,
            error=repr(exc),
            apply_cleanup=apply_cleanup,
        )


def _extract_once(
    pdf_path: str | Path,
    *,
    backend: str,
    requested_backend: str | None = None,
    apply_cleanup: bool = True,
) -> PDFExtractionResult:
    requested_backend = requested_backend or backend
    if backend in {"pypdf", "auto"}:
        return extract_text_pypdf(pdf_path, requested_backend=requested_backend, apply_cleanup=apply_cleanup)
    if backend == "pymupdf":
        return extract_text_pymupdf(pdf_path, requested_backend=requested_backend, columns=False, apply_cleanup=apply_cleanup)
    if backend == "pymupdf_columns":
        return extract_text_pymupdf(pdf_path, requested_backend=requested_backend, columns=True, apply_cleanup=apply_cleanup)
    raise ValueError(f"Unsupported PDF backend: {backend}")


def extract_pdf_text(
    pdf_path: str | Path,
    *,
    backend: str = "pymupdf_columns",
    fallback_backend: str | None = "pypdf",
    use_fallback: bool = True,
    suspicious_min_clean_chars: int = 1000,
    apply_cleanup: bool = True,
) -> PDFExtractionResult:
    """
    Extract and cleanup PDF text, optionally trying a fallback backend.

    If fallback is enabled and the primary extraction is suspicious, the fallback
    result is used when it is not suspicious or when it extracts more cleaned text.
    """
    primary_backend = "pymupdf_columns" if backend == "auto" else backend
    primary = _extract_once(
        pdf_path,
        backend=primary_backend,
        requested_backend=backend,
        apply_cleanup=apply_cleanup,
    )

    if not use_fallback or fallback_backend is None:
        return primary

    if fallback_backend == primary.backend_used:
        return primary

    primary_suspicious = is_suspicious_extraction(
        primary,
        min_clean_chars=suspicious_min_clean_chars,
    )

    if not primary_suspicious:
        return primary

    fallback = _extract_once(
        pdf_path,
        backend=fallback_backend,
        requested_backend=backend,
        apply_cleanup=apply_cleanup,
    )
    fallback_suspicious = is_suspicious_extraction(
        fallback,
        min_clean_chars=suspicious_min_clean_chars,
    )

    if (not fallback_suspicious) or (fallback.clean_chars > primary.clean_chars):
        fallback.fallback_used = True
        if primary.error:
            fallback.error = f"primary {primary.backend_used}: {primary.status}: {primary.error} || fallback: {fallback.error}"
        else:
            fallback.error = f"primary {primary.backend_used}: {primary.status}; fallback selected"
        return fallback

    return primary
