from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable
import hashlib
from collections import defaultdict


# ----------------------------
# General helpers
# ----------------------------

def local_name(tag: str) -> str:
    """Remove XML namespace from tag name."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_title(title: str | None) -> str | None:
    if not title:
        return None

    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title or None


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None

    doi = doi.strip().lower()
    doi = doi.removeprefix("doi:")
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.strip(" .;,)")
    return doi or None


def extract_doi_from_url(url: str | None) -> str | None:
    """
    Extract DOI from URLs like:
    https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/sstr.202000144
    https://doi.org/10.xxxx/...
    """
    if not url:
        return None

    m = re.search(r"(10\.\d{4,9}/[^\s?#]+)", url, flags=re.I)
    if not m:
        return None

    doi = m.group(1)

    # common cleanups
    doi = re.sub(r"\.pdf$", "", doi, flags=re.I)
    doi = doi.strip("/")

    return normalize_doi(doi)


def journal_name(record: dict) -> str:
    journal = record.get("journal") or {}
    name = journal.get("name")

    if not name:
        return "UNKNOWN"

    return str(name).strip() or "UNKNOWN"


def write_counter_csv(counter: Counter, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["journal", "count"])

        for journal, count in counter.most_common():
            writer.writerow([journal, count])


# ----------------------------
# JSONL helpers
# ----------------------------

def iter_jsonl(jsonl_path: str | Path) -> Iterable[dict]:
    jsonl_path = Path(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON at line {line_no}: {e}")


def make_pdf_article_keys(record: dict) -> set[tuple[str, str]]:
    """
    Article identity keys for PDF metadata.

    Preferred:
      ("doi", normalized_doi)

    Fallback:
      ("title_year", normalized_title|year)
    """
    keys: set[tuple[str, str]] = set()

    year = record.get("year")
    title = record.get("title")

    doi = normalize_doi(record.get("doi"))

    if doi is None:
        pdf_url = (record.get("openAccessPdf") or {}).get("url")
        doi = extract_doi_from_url(pdf_url)

    if doi:
        keys.add(("doi", doi))

    title_key = normalize_title(title)
    if title_key and year:
        keys.add(("title_year", f"{title_key}|{year}"))

    return keys


# ----------------------------
# PDF helpers
# ----------------------------

def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Hash full file content.
    Used only after size prefilter.
    """
    path = Path(path)
    h = hashlib.blake2b(digest_size=32)

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def find_duplicate_pdfs_with_size_prefilter(
    pdf_dir: str | Path,
    recursive: bool = False,
) -> dict[str, list[Path]]:
    """
    Find exact duplicate PDFs.

    Fast logic:
      1. group by file size
      2. hash only files from groups with the same size
      3. return only real duplicate groups by hash

    Returns:
      {
        hash: [path1, path2, ...],
        ...
      }
    """
    pdf_dir = Path(pdf_dir)
    pattern = "**/*.pdf" if recursive else "*.pdf"

    pdf_paths = sorted(pdf_dir.glob(pattern))

    by_size: dict[int, list[Path]] = defaultdict(list)

    for path in pdf_paths:
        try:
            size = path.stat().st_size
            by_size[size].append(path.resolve())
        except OSError as e:
            print(f"[WARN] Cannot stat PDF {path}: {e}")

    same_size_groups = {
        size: paths
        for size, paths in by_size.items()
        if len(paths) > 1
    }

    files_to_hash = sum(len(paths) for paths in same_size_groups.values())

    print(f"PDF files total: {len(pdf_paths)}")
    print(f"PDF size-collision groups: {len(same_size_groups)}")
    print(f"PDF files requiring hash: {files_to_hash}")

    by_hash: dict[str, list[Path]] = defaultdict(list)

    for size, paths in same_size_groups.items():
        for path in paths:
            try:
                h = hash_file(path)
                by_hash[h].append(path)
            except OSError as e:
                print(f"[WARN] Cannot hash PDF {path}: {e}")

    duplicate_groups = {
        h: paths
        for h, paths in by_hash.items()
        if len(paths) > 1
    }

    duplicate_files_total = sum(len(paths) for paths in duplicate_groups.values())

    print(f"Real duplicate PDF groups: {len(duplicate_groups)}")
    print(f"Real duplicate PDF files total: {duplicate_files_total}")

    return duplicate_groups


def build_duplicate_pdf_skip_set_with_size_prefilter(
    pdf_dir: str | Path,
    recursive: bool = False,
) -> set[Path]:
    """
    Build a set of duplicate PDF paths to skip.

    Keeps the first file from each exact duplicate group,
    skips all others.
    """
    duplicate_groups = find_duplicate_pdfs_with_size_prefilter(
        pdf_dir=pdf_dir,
        recursive=recursive,
    )

    skip_paths: set[Path] = set()

    for h, paths in duplicate_groups.items():
        paths = sorted(paths)

        # keep paths[0], skip the rest
        skip_paths.update(paths[1:])

    print(f"PDF duplicate files to skip: {len(skip_paths)}")

    return skip_paths

def build_pdf_index(pdf_dir: str | Path) -> dict[str, Path]:
    """
    Index PDFs by lowercase stem.
    Example:
      00000abc.pdf -> {"00000abc": Path(...)}
    """
    pdf_dir = Path(pdf_dir)

    pdf_index: dict[str, Path] = {}

    for path in pdf_dir.glob("*.pdf"):
        pdf_index[path.stem.lower()] = path

    for path in pdf_dir.glob("*.PDF"):
        pdf_index[path.stem.lower()] = path

    return pdf_index

def collect_pdf_paths(
    pdf_input: str | Path | Iterable[str | Path] | None,
    recursive: bool = True,
) -> list[Path]:
    """
    Collect PDF paths from:
      - one PDF file
      - one directory
      - list of PDF files/directories

    For directories, searches recursively by default.
    """
    if pdf_input is None:
        return []

    if isinstance(pdf_input, (str, Path)):
        items = [Path(pdf_input)]
    else:
        items = [Path(x) for x in pdf_input]

    pdf_paths: list[Path] = []

    for item in items:
        if item.is_dir():
            pattern = "**/*" if recursive else "*"
            pdf_paths.extend(
                p
                for p in item.glob(pattern)
                if p.is_file() and p.suffix.lower() == ".pdf"
            )
        elif item.is_file() and item.suffix.lower() == ".pdf":
            pdf_paths.append(item)
        else:
            print(f"[WARN] Extra PDF input not found or not PDF: {item}")

    return sorted({p.resolve() for p in pdf_paths}, key=lambda p: p.as_posix())

def can_open_pdf(pdf_path: str | Path) -> tuple[bool, int | None, str | None]:
    """
    Returns:
      ok, page_count, error_message

    Requires:
      pip install pypdf

    Notes:
      Some PDFs are technically encrypted but can be opened with an empty
      password. In pypdf, reader.is_encrypted may remain True even after a
      successful decrypt(""). Therefore, do not reject the PDF only because
      is_encrypted is still True after decrypt. Instead, try to access pages.
    """
    pdf_path = Path(pdf_path)

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Please install pypdf first: pip install pypdf")

    try:
        reader = PdfReader(str(pdf_path), strict=False)

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as e:
                return False, None, f"encrypted_or_decrypt_failed: {e}"

            # pypdf usually returns 0 for failure, 1 or 2 for success.
            # Some versions may return PasswordType enum-like values.
            if decrypt_result == 0:
                return False, None, "encrypted_pdf_password_required"

        try:
            page_count = len(reader.pages)
        except Exception as e:
            return False, None, f"cannot_read_pages_after_decrypt: {e}"

        if page_count <= 0:
            return False, page_count, "zero_pages"

        # Force reading the first page object.
        try:
            _ = reader.pages[0].mediabox
        except Exception as e:
            return False, page_count, f"cannot_read_first_page: {e}"

        return True, page_count, None

    except Exception as e:
        return False, None, str(e)
        
# ----------------------------
# Language helpers
# ----------------------------

ENGLISH_LANGUAGE_VALUES = {"en", "eng", "english"}


def normalize_language_value(value) -> str | None:
    """
    Normalize a possible language value from article metadata.

    Supports:
      "en", "eng", "English"
      ["en"]
      {"code": "en"} / {"name": "English"}
    """
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("code", "name", "language", "lang"):
            normalized = normalize_language_value(value.get(key))
            if normalized:
                return normalized
        return None

    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_language_value(item)
            if normalized:
                return normalized
        return None

    value = str(value).strip().lower()
    return value or None


def extract_record_language(record: dict) -> str | None:
    """
    Extract declared language from common metadata fields, if present.

    Many Semantic Scholar records do not contain a language field, so this
    function often returns None. In that case we fall back to a text heuristic.
    """
    for key in (
        "language",
        "lang",
        "languageCode",
        "language_code",
        "detectedLanguage",
        "detected_language",
    ):
        value = normalize_language_value(record.get(key))
        if value:
            return value

    return None


def is_strong_non_latin_letter(ch: str) -> bool:
    """
    Return True for scripts that are a strong signal that the article is not
    English-language full text.

    Greek symbols are intentionally not counted here because materials-science
    English papers frequently contain a, ß, ?, µ, etc.
    """
    code = ord(ch)

    return (
        0x0400 <= code <= 0x04FF  # Cyrillic
        or 0x0590 <= code <= 0x05FF  # Hebrew
        or 0x0600 <= code <= 0x06FF  # Arabic
        or 0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul
    )


ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "in", "to", "for", "with", "on", "by", "as",
    "is", "are", "was", "were", "be", "been", "from", "this", "that",
    "these", "those", "we", "it", "an", "a", "or", "at", "which",
    "can", "using", "used", "use", "between", "into", "than", "their",
    "our", "such", "result", "results", "study", "show", "shown",
}


def looks_english_text(text: str, min_letters: int = 80) -> tuple[bool, str]:
    """
    Lightweight no-dependency English-language heuristic.

    This is deliberately conservative:
      - reject texts dominated by strong non-Latin scripts;
      - require mostly Latin letters;
      - require a small proportion of common English function words.

    It is intended for selecting corpus candidates, not for perfect language
    identification.
    """
    text = text or ""
    letters = [ch for ch in text if ch.isalpha()]

    if len(letters) < min_letters:
        return False, f"too_few_letters:{len(letters)}"

    strong_non_latin = sum(1 for ch in letters if is_strong_non_latin_letter(ch))
    strong_non_latin_ratio = strong_non_latin / len(letters)

    if strong_non_latin_ratio > 0.05:
        return False, f"strong_non_latin_ratio:{strong_non_latin_ratio:.3f}"

    latin_letters = sum(
        1
        for ch in letters
        if ("a" <= ch.lower() <= "z")
    )
    latin_ratio = latin_letters / len(letters)

    if latin_ratio < 0.80:
        return False, f"latin_letter_ratio:{latin_ratio:.3f}"

    tokens = re.findall(r"[a-z]+", text.lower())

    if len(tokens) < 30:
        return False, f"too_few_latin_tokens:{len(tokens)}"

    english_word_hits = sum(1 for token in tokens if token in ENGLISH_FUNCTION_WORDS)
    english_word_ratio = english_word_hits / len(tokens)

    if english_word_ratio < 0.03:
        return False, f"english_function_word_ratio:{english_word_ratio:.3f}"

    return True, (
        f"heuristic_ok:"
        f"latin_ratio={latin_ratio:.3f};"
        f"strong_non_latin_ratio={strong_non_latin_ratio:.3f};"
        f"english_function_word_ratio={english_word_ratio:.3f}"
    )


def extract_pdf_text_sample_for_language(
    pdf_path: str | Path,
    max_pages: int = 2,
    max_chars: int = 8000,
) -> str:
    """
    Extract a small text sample from the first pages for language checking.

    Tries PyMuPDF first because it usually handles scientific PDFs better.
    Falls back to pypdf to avoid adding a hard dependency.
    """
    pdf_path = Path(pdf_path)

    # Prefer PyMuPDF if available.
    try:
        import fitz  # PyMuPDF

        parts: list[str] = []

        with fitz.open(str(pdf_path)) as doc:
            for page in doc[:max_pages]:
                parts.append(page.get_text("text", sort=True) or "")
                if sum(len(p) for p in parts) >= max_chars:
                    break

        text = "\n".join(parts).strip()

        if text:
            return text[:max_chars]

    except Exception:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path), strict=False)
        parts = []

        for page in reader.pages[:max_pages]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue

            if sum(len(p) for p in parts) >= max_chars:
                break

        return "\n".join(parts).strip()[:max_chars]

    except Exception:
        return ""


def is_english_pdf_article(
    record: dict,
    pdf_path: str | Path,
    max_pages: int = 2,
) -> tuple[bool, str, int]:
    """
    Decide whether an article should be treated as English-language.

    Returns:
      ok, reason, sample_chars
    """
    declared_language = extract_record_language(record)

    if declared_language:
        if declared_language in ENGLISH_LANGUAGE_VALUES:
            return True, f"metadata_language:{declared_language}", 0

        return False, f"metadata_language:{declared_language}", 0

    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    pdf_sample = extract_pdf_text_sample_for_language(pdf_path, max_pages=max_pages)

    detection_text = "\n".join([title, abstract, pdf_sample]).strip()
    ok, reason = looks_english_text(detection_text)

    return ok, reason, len(pdf_sample)



# ----------------------------
# XML helpers
# ----------------------------

def collect_xml_paths(xml_inputs: str | Path | Iterable[str | Path]) -> list[Path]:
    """
    Accepts:
      - one XML file
      - one directory
      - list of XML files/directories
    """
    if isinstance(xml_inputs, (str, Path)):
        items = [Path(xml_inputs)]
    else:
        items = [Path(x) for x in xml_inputs]

    xml_paths: list[Path] = []

    for item in items:
        if item.is_dir():
            xml_paths.extend(sorted(item.glob("**/*.xml")))
        elif item.is_file() and item.suffix.lower() == ".xml":
            xml_paths.append(item)
        else:
            print(f"[WARN] XML input not found or not XML: {item}")

    return xml_paths


def extract_xml_doi(root: ET.Element) -> str | None:
    for el in root.iter():
        name = local_name(el.tag).lower()
        text = (el.text or "").strip()

        if not text:
            continue

        if name == "doi":
            return normalize_doi(text)

        if name == "identifier" and "doi:" in text.lower():
            return normalize_doi(text.lower().split("doi:", 1)[1])

    return None


def extract_xml_title(root: ET.Element) -> str | None:
    """
    For Elsevier XML, dc:title is usually near the top.
    If not found, falls back to the first title-like tag.
    """
    for el in root.iter():
        name = local_name(el.tag).lower()

        if name == "title":
            text = " ".join(" ".join(el.itertext()).split())
            if text:
                return text

    return None


def extract_xml_year(root: ET.Element) -> int | None:
    year_tags = {
        "cover-date-year",
        "year-nav",
        "publication-year",
        "copyright-year",
    }

    date_tags = {
        "coverdate",
        "coverdisplaydate",
        "publicationdate",
        "date-search-begin",
        "date-search-end",
        "cover-date-start",
        "cover-date-end",
    }

    for el in root.iter():
        name = local_name(el.tag).lower()
        text = (el.text or "").strip()

        if name in year_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return int(m.group())

    for el in root.iter():
        name = local_name(el.tag).lower()
        text = (el.text or "").strip()

        if name in date_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return int(m.group())

    # fallback: attributes like yyyymmdd="20170627"
    for el in root.iter():
        for value in el.attrib.values():
            m = re.search(r"\b(19|20)\d{2}\b", str(value))
            if m:
                return int(m.group())

    return None


def make_xml_article_keys(xml_path: str | Path) -> set[tuple[str, str]]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    keys: set[tuple[str, str]] = set()

    doi = extract_xml_doi(root)
    title = extract_xml_title(root)
    year = extract_xml_year(root)

    if doi:
        keys.add(("doi", doi))

    title_key = normalize_title(title)
    if title_key and year:
        keys.add(("title_year", f"{title_key}|{year}"))

    return keys


def build_xml_article_key_set(xml_inputs: str | Path | Iterable[str | Path]) -> set[tuple[str, str]]:
    xml_paths = collect_xml_paths(xml_inputs)

    keys: set[tuple[str, str]] = set()

    for xml_path in xml_paths:
        try:
            keys.update(make_xml_article_keys(xml_path))
        except Exception as e:
            print(f"[WARN] Failed to read XML metadata from {xml_path}: {e}")

    print(f"XML files scanned: {len(xml_paths)}")
    print(f"XML article keys collected: {len(keys)}")

    return keys


# ----------------------------
# Main selection function
# ----------------------------


def select_valid_pdf_articles(
    jsonl_path: str | Path,
    pdf_dir: str | Path,
    xml_inputs: str | Path | Iterable[str | Path],
    output_titles_path: str | Path,
    output_selected_jsonl_path: str | Path | None = None,
    output_stats_dir: str | Path = "pdf_selection_stats",
    min_year: int = 1991,
    remove_duplicate_pdf_files: bool = True,
    require_english_language: bool = True,
    language_check_max_pages: int = 2,
    extra_pdf_dir: str | Path | None = None,
) -> list[dict]:
    """
    Select PDF articles that:
      - are present in pdf_dir as paperId.pdf
      - open correctly
      - have year >= min_year
      - are likely English-language articles, if require_english_language=True
      - do not duplicate existing XML articles by DOI or title+year
      - additionally include all valid PDFs from extra_pdf_dir, if provided

    Saves:
      - titles of selected PDF articles
      - journal stats for found/openable/selected PDFs
      - optional selected records JSONL
    """
    jsonl_path = Path(jsonl_path)
    pdf_dir = Path(pdf_dir)
    output_titles_path = Path(output_titles_path)
    output_stats_dir = Path(output_stats_dir)

    xml_keys = build_xml_article_key_set(xml_inputs)
    pdf_index = build_pdf_index(pdf_dir)

    if remove_duplicate_pdf_files:
        duplicate_pdf_skip_set = build_duplicate_pdf_skip_set_with_size_prefilter(
            pdf_dir=pdf_dir,
            recursive=False,
        )
    else:
        duplicate_pdf_skip_set = set()
    
    found_pdf_journals = Counter()
    openable_pdf_journals = Counter()
    selected_pdf_journals = Counter()

    selected_records: list[dict] = []

    total_jsonl_records = 0
    missing_pdf = 0
    broken_pdf = 0
    skipped_old = 0
    skipped_xml_duplicate = 0
    skipped_duplicate_pdf_file = 0
    skipped_non_english = 0
    selected = 0
    
    extra_pdf_total = 0
    extra_pdf_selected = 0
    extra_pdf_broken = 0
    extra_pdf_skipped_duplicate_path = 0
    extra_pdf_skipped_non_english = 0

    broken_rows = []
    non_english_rows = []

    for record in iter_jsonl(jsonl_path):
        total_jsonl_records += 1

        paper_id = record.get("paperId")
        title = record.get("title") or ""
        year = record.get("year")
        journal = journal_name(record)

        if not paper_id:
            missing_pdf += 1
            continue

        pdf_path = pdf_index.get(str(paper_id).lower())

        if pdf_path is None:
            missing_pdf += 1
            continue
        pdf_path = pdf_path.resolve()

        if pdf_path in duplicate_pdf_skip_set:
            skipped_duplicate_pdf_file += 1
            continue
                

        found_pdf_journals[journal] += 1

        ok, page_count, error = can_open_pdf(pdf_path)

        if not ok:
            broken_pdf += 1
            broken_rows.append(
                {
                    "paperId": paper_id,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "pdf_path": str(pdf_path),
                    "error": error,
                }
            )
            continue

        openable_pdf_journals[journal] += 1

        if year is None or int(year) < min_year:
            skipped_old += 1
            continue

        pdf_keys = make_pdf_article_keys(record)

        if pdf_keys and (pdf_keys & xml_keys):
            skipped_xml_duplicate += 1
            continue

        if require_english_language:
            is_english, language_reason, language_sample_chars = is_english_pdf_article(
                record=record,
                pdf_path=pdf_path,
                max_pages=language_check_max_pages,
            )

            if not is_english:
                skipped_non_english += 1
                non_english_rows.append(
                    {
                        "paperId": paper_id,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "pdf_path": str(pdf_path),
                        "language_reason": language_reason,
                        "language_sample_chars": language_sample_chars,
                    }
                )
                continue

        selected_pdf_journals[journal] += 1
        selected += 1

        selected_record = dict(record)
        selected_record["_pdf_path"] = str(pdf_path)
        selected_record["_page_count"] = page_count
        selected_records.append(selected_record)

    # Optionally add all PDFs from an extra folder.
    #
    # These PDFs do not need to be present in the JSONL metadata.
    # They are added as synthetic records with _pdf_path, which is enough
    # for the downstream corpus-building script.
    if extra_pdf_dir is not None:
        extra_pdf_paths = collect_pdf_paths(extra_pdf_dir, recursive=True)

        print(f"Extra PDF files found: {len(extra_pdf_paths)}")

        already_selected_paths = {
            Path(record["_pdf_path"]).resolve()
            for record in selected_records
            if record.get("_pdf_path")
        }

        for pdf_path in extra_pdf_paths:
            extra_pdf_total += 1
            pdf_path = pdf_path.resolve()

            if pdf_path in already_selected_paths:
                extra_pdf_skipped_duplicate_path += 1
                continue

            journal = "EXTRA_PDF_DIR"
            title = pdf_path.stem
            paper_id = f"extra_pdf::{pdf_path.stem}"

            found_pdf_journals[journal] += 1

            ok, page_count, error = can_open_pdf(pdf_path)

            if not ok:
                extra_pdf_broken += 1
                broken_pdf += 1
                broken_rows.append(
                    {
                        "paperId": paper_id,
                        "title": title,
                        "journal": journal,
                        "year": None,
                        "pdf_path": str(pdf_path),
                        "error": error,
                    }
                )
                continue

            openable_pdf_journals[journal] += 1

            if require_english_language:
                synthetic_record = {
                    "paperId": paper_id,
                    "title": title,
                    "year": None,
                    "journal": {"name": journal},
                    "_source": "extra_pdf_dir",
                }

                is_english, language_reason, language_sample_chars = is_english_pdf_article(
                    record=synthetic_record,
                    pdf_path=pdf_path,
                    max_pages=language_check_max_pages,
                )

                if not is_english:
                    extra_pdf_skipped_non_english += 1
                    skipped_non_english += 1
                    non_english_rows.append(
                        {
                            "paperId": paper_id,
                            "title": title,
                            "journal": journal,
                            "year": None,
                            "pdf_path": str(pdf_path),
                            "language_reason": language_reason,
                            "language_sample_chars": language_sample_chars,
                        }
                    )
                    continue

            selected_pdf_journals[journal] += 1
            selected += 1
            extra_pdf_selected += 1

            selected_record = {
                "paperId": paper_id,
                "title": title,
                "year": None,
                "journal": {"name": journal},
                "_source": "extra_pdf_dir",
                "_pdf_path": str(pdf_path),
                "_page_count": page_count,
            }

            selected_records.append(selected_record)
            already_selected_paths.add(pdf_path)

    # Save selected titles
    output_titles_path.parent.mkdir(parents=True, exist_ok=True)

    with output_titles_path.open("w", encoding="utf-8") as f:
        for record in selected_records:
            f.write((record.get("title") or "").strip() + "\n")

    # Optional: save selected full metadata
    if output_selected_jsonl_path is not None:
        output_selected_jsonl_path = Path(output_selected_jsonl_path)
        output_selected_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with output_selected_jsonl_path.open("w", encoding="utf-8") as f:
            for record in selected_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save journal statistics
    output_stats_dir.mkdir(parents=True, exist_ok=True)

    write_counter_csv(
        found_pdf_journals,
        output_stats_dir / "journal_stats_found_pdfs.csv",
    )

    write_counter_csv(
        openable_pdf_journals,
        output_stats_dir / "journal_stats_openable_pdfs.csv",
    )

    write_counter_csv(
        selected_pdf_journals,
        output_stats_dir / "journal_stats_selected_pdfs.csv",
    )

    # Save broken PDF report
    broken_report_path = output_stats_dir / "broken_pdfs.csv"

    with broken_report_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["paperId", "title", "journal", "year", "pdf_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(broken_rows)

    # Save non-English / unknown-language report
    non_english_report_path = output_stats_dir / "non_english_or_unknown_language_pdfs.csv"

    with non_english_report_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paperId",
            "title",
            "journal",
            "year",
            "pdf_path",
            "language_reason",
            "language_sample_chars",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(non_english_rows)

    # Save summary
    summary = {
        "total_jsonl_records": total_jsonl_records,
        "pdf_files_indexed": len(pdf_index),
        "missing_pdf": missing_pdf,
        "broken_pdf": broken_pdf,
        "skipped_old_or_unknown_year": skipped_old,
        "skipped_xml_duplicate": skipped_xml_duplicate,
        "skipped_duplicate_pdf_file": skipped_duplicate_pdf_file,
        "skipped_non_english_or_unknown_language": skipped_non_english,
        "selected": selected,
        
        "extra_pdf_dir": str(extra_pdf_dir) if extra_pdf_dir is not None else None,
        "extra_pdf_total": extra_pdf_total,
        "extra_pdf_selected": extra_pdf_selected,
        "extra_pdf_broken": extra_pdf_broken,
        "extra_pdf_skipped_duplicate_path": extra_pdf_skipped_duplicate_path,
        "extra_pdf_skipped_non_english": extra_pdf_skipped_non_english,
        
        "min_year": min_year,
        "require_english_language": require_english_language,
        "language_check_max_pages": language_check_max_pages,
        "output_titles_path": str(output_titles_path),
        "output_stats_dir": str(output_stats_dir),
    }

    with (output_stats_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return selected_records
    
    
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select valid PDF articles and write an auditable selection manifest."
    )
    parser.add_argument("--jsonl", required=True, help="Input bibliographic JSONL.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing paperId.pdf files.")
    parser.add_argument(
        "--xml-inputs", nargs="*", default=[],
        help="XML files/directories used to exclude cross-source duplicates.",
    )
    parser.add_argument("--output-titles", required=True)
    parser.add_argument("--output-selected-jsonl", default=None)
    parser.add_argument("--output-stats-dir", required=True)
    parser.add_argument("--min-year", type=int, default=1991)
    parser.add_argument("--extra-pdf-dir", default=None)
    parser.add_argument("--keep-duplicate-pdf-files", action="store_true")
    parser.add_argument("--no-language-filter", action="store_true")
    parser.add_argument("--language-check-max-pages", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    select_valid_pdf_articles(
        jsonl_path=args.jsonl,
        pdf_dir=args.pdf_dir,
        xml_inputs=args.xml_inputs,
        output_titles_path=args.output_titles,
        output_selected_jsonl_path=args.output_selected_jsonl,
        output_stats_dir=args.output_stats_dir,
        min_year=args.min_year,
        remove_duplicate_pdf_files=not args.keep_duplicate_pdf_files,
        require_english_language=not args.no_language_filter,
        language_check_max_pages=args.language_check_max_pages,
        extra_pdf_dir=args.extra_pdf_dir,
    )


if __name__ == "__main__":
    main()
