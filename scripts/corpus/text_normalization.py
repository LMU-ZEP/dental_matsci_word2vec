"""
Scientific-text normalization for dental/materials-science corpora.

The module deliberately separates PDF-specific cleanup from XML-safe cleanup:

- PDF text is layout-derived and often contains line-break hyphenation,
  attached citation markers, PDF font/control-character artifacts, and broken
  mathematical symbols.
- XML text is usually logical text. It should not go through PDF hyphenation
  repair. For XML, we remove only structured bracketed numeric citations by
  default, e.g. [4,5] or [75-78].
"""

from __future__ import annotations

import re
import unicodedata
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

# Pairs for which a hyphen across a line break is likely meaningful and should be
# preserved, not removed. This is deliberately short and domain-oriented.
PRESERVE_HYPHEN_LINEBREAK_PAIRS = {
    ("3y", "tzp"),
    ("4y", "tzp"),
    ("5y", "tzp"),
    ("3y", "psz"),
    ("4y", "psz"),
    ("5y", "psz"),
    ("y", "tzp"),
    ("y", "psz"),
    ("short", "fiber"),
    ("glass", "fiber"),
    ("carbon", "fiber"),
    ("fiber", "reinforced"),
    ("resin", "based"),
    ("zirconia", "based"),
    ("silicate", "based"),
    ("calcium", "silicate"),
    ("tricalcium", "silicate"),
    ("sol", "gel"),
    ("field", "cooled"),
    ("zero", "field"),
}

# Numeric citation grammar used by both XML and PDF citation cleanup.
# Examples: 4, 4-5, 4–5, 4,5, 4, 5, 7-9.
_CITATION_ITEM = r"\d{1,4}(?:\s*[-–—]\s*\d{1,4})?"
_CITATION_LIST = rf"{_CITATION_ITEM}(?:\s*,\s*{_CITATION_ITEM})*"


def normalize_scientific_unicode(text: str) -> str:
    """
    Normalize Unicode variants relevant for scientific text.

    Examples
    --------
    H₂O -> H2O
    Ca(OH)₂ -> Ca(OH)2
    4Y–PSZ -> 4Y-PSZ
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(DASH_TRANSLATION)
    text = text.translate(SUBSCRIPT_SUPERSCRIPT_TRANSLATION)
    return text


def remove_surrogates(text: str) -> str:
    """Remove invalid standalone Unicode surrogate code points."""
    return re.sub(r"[\ud800-\udfff]", "", text)


def remove_control_characters(text: str, *, keep_newlines: bool = True) -> str:
    """
    Remove Unicode control/format characters that commonly appear as PDF
    extraction artifacts.

    Parameters
    ----------
    keep_newlines:
        Keep \n/\r/\t while PDF hyphenation repair still needs line breaks.
        Later whitespace normalization will collapse them.
    """
    keep = {"\n", "\r", "\t"} if keep_newlines else set()
    out_chars: list[str] = []

    for ch in text:
        if ch in keep:
            out_chars.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in {"Cc", "Cf"}:
            out_chars.append(" ")
        else:
            out_chars.append(ch)

    return "".join(out_chars)


def clean_object(obj: Any) -> Any:
    """Recursively remove Unicode surrogates from strings in nested objects."""
    if isinstance(obj, str):
        return remove_surrogates(obj)
    if isinstance(obj, list):
        return [clean_object(item) for item in obj]
    if isinstance(obj, dict):
        return {clean_object(k): clean_object(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Citation cleanup
# ---------------------------------------------------------------------------


def count_xml_numeric_citations(text: str) -> int:
    """
    Count XML-style structured bracketed numeric citations.

    Examples counted:
      [4]
      [4,5]
      [11, 12]
      [75-78]
    """
    pattern = rf"\[\s*{_CITATION_LIST}\s*\]"
    return len(re.findall(pattern, text))


def remove_xml_numeric_citations(text: str) -> str:
    """
    Remove only structured bracketed numeric citation markers from XML text.

    Removes:
      [4]
      [4,5]
      [11, 12]
      [75-78]

    Does not remove scientific/experimental numbers such as:
      PEI25k, Cu 2+, 25 kDa, 3-(4,5-dimethyl...), 100 Hz.
    """
    pattern = rf"\[\s*{_CITATION_LIST}\s*\]"
    return re.sub(pattern, " ", text)


def count_pdf_inline_citation_markers(text: str) -> int:
    """
    Approximate count of common PDF-derived numeric citation markers.

    This is intended for diagnostics/manifests, not bibliometric analysis.
    """
    patterns = [
        # Word immediately followed by citations: apicoectomy70,71)
        r"\b[A-Za-z]{4,}\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
        # Bracketed references: [70], [70,71], [75-78]
        rf"\[\s*{_CITATION_LIST}\s*\]",
        # Parenthesized references: (70), (70, 71), (75-78)
        rf"\(\s*{_CITATION_LIST}\s*\)",
        # Sentence-final citations glued after punctuation: state.4,14 The
        r"[A-Za-z]\.\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Z])",
        # Citations after author abbreviation: al.,30 explicitly
        r"[A-Za-z]\.\s*,\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Za-z])",
        # Double-comma citation artifacts: interest,,30 and
        r"[A-Za-z],{2,}\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Za-z])",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def remove_pdf_inline_citations(text: str) -> str:
    """
    Remove common numeric citation markers from PDF-extracted scientific text.

    Handles examples:
      apicoectomy70,71) -> apicoectomy
      pulp capping72) -> pulp capping
      biocompatibility75-78) -> biocompatibility
      [70,71] -> removed
      (70, 71) -> removed
      state.4,14 The -> state. The
      al.,30 explicitly -> al. explicitly
      interest,,30 and -> interest, and

    The patterns deliberately target citation-like short numeric groups and avoid
    removing formula/material tokens such as H2O, Ca(OH)2, Pr0.7Ca0.3MnO3, or
    3Y-TZP.
    """
    # XML-style bracketed citations are also common in PDF extraction.
    text = remove_xml_numeric_citations(text)

    # Long alphabetic word immediately followed by citation numbers and a
    # closing parenthesis. The long-word constraint avoids damaging formulae
    # such as H2O2) or short abbreviations.
    text = re.sub(
        r"\b([A-Za-z]{4,})\s*\d{1,4}(?:\s*[-,]\s*\d{1,4})*\s*\)",
        r"\1",
        text,
    )

    # Parenthesized citation lists.
    text = re.sub(
        rf"\(\s*{_CITATION_LIST}\s*\)",
        " ",
        text,
    )

    # Sentence-final citations glued after a full stop: state.4,14 The -> state. The.
    text = re.sub(
        r"([A-Za-z]\.)\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Z])",
        r"\1",
        text,
    )

    # Author/reference style artifact: al.,30 explicitly -> al. explicitly.
    text = re.sub(
        r"([A-Za-z]\.)\s*,\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Za-z])",
        r"\1",
        text,
    )

    # Double comma artifact: interest,,30 and -> interest, and.
    text = re.sub(
        r"([A-Za-z]),{2,}\s*\d{1,4}(?:\s*[,;-]\s*\d{1,4})*(?=\s+[A-Za-z])",
        r"\1,",
        text,
    )

    return text


# Backward-compatible names used by pdf_extraction.py.
def count_inline_citation_markers(text: str) -> int:
    return count_pdf_inline_citation_markers(text)


def remove_inline_citations(text: str) -> str:
    return remove_pdf_inline_citations(text)


# ---------------------------------------------------------------------------
# PDF hyphenation repair
# ---------------------------------------------------------------------------


def _repair_hyphen_linebreak(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    pair = (left.lower(), right.lower())

    if pair in PRESERVE_HYPHEN_LINEBREAK_PAIRS:
        return f"{left}-{right}"

    # Preserve chemical/materials abbreviations and uppercase code fragments.
    # Examples: 3Y-\nTZP -> 3Y-TZP, Y-\nPSZ -> Y-PSZ.
    if any(ch.isdigit() for ch in left + right) or left.isupper() or right.isupper():
        return f"{left}-{right}"

    # Ordinary lowercase word hyphenation: infor-\nmation -> information.
    if left.islower() and right.islower() and len(left) >= 2 and len(right) >= 2:
        return f"{left}{right}"

    # Safe fallback: preserve the hyphen rather than destroying a scientific token.
    return f"{left}-{right}"


def fix_pdf_hyphenation(text: str) -> str:
    """
    Conservatively repair PDF line-break artifacts.

    - infor-\nmation -> information
    - 3Y-\nTZP -> 3Y-TZP
    - 4Y-\nPSZ -> 4Y-PSZ
    - short-\nfiber -> short-fiber
    - remaining line breaks -> spaces
    """
    text = re.sub(r"\b([A-Za-z0-9]+)-\s*\n\s*([A-Za-z0-9]+)\b", _repair_hyphen_linebreak, text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text


# ---------------------------------------------------------------------------
# High-level cleanup entry points
# ---------------------------------------------------------------------------


def cleanup_pdf_extracted_text(text: str) -> str:
    """
    Full cleanup for PDF-extracted scientific text before NLP preprocessing.

    PDF-specific steps include conservative line-break hyphenation repair and a
    broader citation cleanup for PDF extraction artifacts such as `.4,14` and
    `word70,71)`.
    """
    text = remove_surrogates(text)
    text = normalize_scientific_unicode(text)
    text = remove_control_characters(text, keep_newlines=True)
    text = fix_pdf_hyphenation(text)
    text = remove_pdf_inline_citations(text)
    text = remove_control_characters(text, keep_newlines=False)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cleanup_xml_raw_text(
    text: str,
    *,
    remove_citations: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """
    XML-safe cleanup for XML-derived raw text.

    Unlike PDF cleanup, this intentionally does NOT call fix_pdf_hyphenation().
    By default, it removes only structured bracketed numeric citations such as
    [4,5] and [75-78]. It does not remove PDF-style attached citations such as
    `word70,71)` because those are rarely needed for XML and could affect real
    scientific numeric expressions.
    """
    text = remove_surrogates(text)
    if normalize_unicode:
        text = normalize_scientific_unicode(text)
    text = remove_control_characters(text, keep_newlines=False)
    if remove_citations:
        text = remove_xml_numeric_citations(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Backward-compatible name used by pdf_extraction.py and older pipeline code.
def cleanup_extracted_text(text: str) -> str:
    return cleanup_pdf_extracted_text(text)
