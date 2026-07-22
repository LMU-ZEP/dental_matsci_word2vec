from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable


def fix_pdf_hyphenation(text: str) -> str:
    """Repair common line-break artifacts while retaining word boundaries."""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)\s*\n\s*(\w+)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def _trim_after_last_eof(path: Path) -> bytes:
    """Return PDF bytes truncated after the final %%EOF marker."""
    data = path.read_bytes()
    eof = data.rfind(b"%%EOF")
    if eof < 0:
        raise ValueError(f"No %%EOF marker found in {path}")
    return data[: eof + len(b"%%EOF")]


def _extract_with_pymupdf(path: Path, repair_after_eof: bool) -> list[str]:
    """Extract page text with PyMuPDF, preserving the page reading order."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is not installed.") from exc

    try:
        document = fitz.open(path)
    except Exception:
        if not repair_after_eof:
            raise
        document = fitz.open(stream=_trim_after_last_eof(path), filetype="pdf")

    try:
        pages = []
        for page in document:
            text = page.get_text("text", sort=True) or ""
            if text.strip():
                pages.append(text)
        return pages
    finally:
        document.close()


def _extract_with_pypdf2(path: Path, repair_after_eof: bool) -> list[str]:
    """Extract page text with the PyPDF2 parser used by the previous pipeline."""
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise ImportError("PyPDF2 is not installed.") from exc

    try:
        reader = PdfReader(str(path))
    except Exception:
        if not repair_after_eof:
            raise
        reader = PdfReader(io.BytesIO(_trim_after_last_eof(path)))

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return pages


def extract_pdf_text(
    path: str | Path,
    min_chars: int = 30,
    repair_after_eof: bool = True,
) -> list[str]:
    """
    Extract one cleaned PDF text chunk.

    PyMuPDF is the primary parser. PyPDF2, which was used in the previous
    pipeline, is retained as a fallback. Each parser can retry after trimming
    bytes located after the final ``%%EOF`` marker.
    """
    path = Path(path)
    extractors: tuple[Callable[[Path, bool], list[str]], ...] = (
        _extract_with_pymupdf,
        _extract_with_pypdf2,
    )

    errors: list[str] = []
    pages: list[str] = []
    for extractor in extractors:
        try:
            pages = extractor(path, repair_after_eof)
            if pages:
                break
        except Exception as exc:  # try the fallback parser before failing
            errors.append(f"{extractor.__name__}: {exc!r}")

    if not pages and errors and len(errors) == len(extractors):
        raise RuntimeError(
            f"Both PDF parsers failed for {path}: " + " | ".join(errors)
        )

    cleaned = fix_pdf_hyphenation("\n".join(pages))
    return [cleaned] if len(cleaned) >= min_chars else []
