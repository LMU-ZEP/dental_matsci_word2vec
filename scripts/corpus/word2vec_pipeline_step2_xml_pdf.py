"""
Pipeline for extracting, merging, normalizing, and preprocessing dental and
materials science corpora from XML and PDF sources.

Steps:
1. Extract text from selected PDF files and/or load an existing XML raw corpus.
2. Merge XML and PDF raw corpora when both sources are provided.
3. Normalize and preprocess the corpus: lowercase, tokenize, lemmatize, and
   remove stopwords/punctuation (with optional numeric-token filtering).
4. Save the tokenized corpus as *.preprocessed.json.

Subsequent phrase modeling and Word2Vec training are handled by the separate
scripts under scripts/modeling/.

PDF/text processing notes:
- PDF extraction is delegated to pdf_extraction.py;
- pypdf, PyMuPDF and column-aware PyMuPDF backends are supported;
- scientific-text cleanup is delegated to text_normalization.py;
- PDF cleanup includes Unicode/dash/subscript/superscript normalization,
  control-character removal, conservative hyphenation repair, PDF-style inline
  citation removal and whitespace cleanup;
- XML cleanup is intentionally XML-safe: Unicode/control-character cleanup,
  optional bracketed numeric citation removal, and whitespace cleanup; it does
  not apply PDF hyphenation repair;
- the merged raw corpus is streamed to disk, so the large XML corpus is not
  loaded into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

try:
    import ijson
except ImportError:
    ijson = None
import spacy
from nltk.corpus import stopwords

from pdf_extraction import extract_pdf_text
import text_normalization as textnorm


DEFAULT_SEED = 42
DEFAULT_SUFFIXES = (".pdf",)


def require_ijson() -> None:
    if ijson is None:
        raise RuntimeError(
            "Missing required dependency 'ijson'. Install repository requirements before running data processing."
        )


# --- NLP setup ----------------------------------------------------------------

# Loaded lazily after CLI parsing so ``--help`` works even before NLP resources
# are installed. Runtime preprocessing behavior is unchanged once initialized.
nlp = None
stop_words: set[str] = set()


def initialize_nlp_resources() -> None:
    global nlp, stop_words
    if nlp is not None:
        return
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise OSError(
            "SpaCy model 'en_core_web_sm' is not installed. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError as exc:
        raise LookupError(
            "NLTK English stopwords are not installed. "
            "Run: python -m nltk.downloader stopwords"
        ) from exc
    nlp.max_length = 10_000_000


# --- Reproducibility utilities ------------------------------------------------


def collect_files(
    input_dirs: Iterable[str | Path],
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
) -> list[Path]:
    """
    Recursively collect files from one or more directories in deterministic order.

    Parameters
    ----------
    input_dirs:
        Directories to search recursively.
    suffixes:
        File suffixes to include. Comparison is case-insensitive.

    Returns
    -------
    list[Path]
        Sorted unique file paths.
    """
    normalized_suffixes = tuple(s.lower() for s in suffixes)
    all_files: set[Path] = set()

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        for path in input_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in normalized_suffixes:
                all_files.add(path)

    return sorted(all_files, key=lambda p: p.as_posix())


def select_random_files(
    files: Iterable[str | Path],
    percentage: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """
    Select a deterministic random subset of files.

    The input files are sorted before sampling, and the selected files are sorted
    again after sampling so downstream processing order is deterministic.

    Parameters
    ----------
    files:
        Input file paths.
    percentage:
        Fraction of files to keep. If >= 1.0, all files are returned.
    seed:
        Seed for the local random generator.

    Returns
    -------
    list[Path]
        Deterministically ordered selected file paths.
    """
    files = [Path(f) for f in files]
    files = sorted(files, key=lambda p: p.as_posix())

    if not files:
        return []

    if percentage >= 1.0:
        return files

    if percentage <= 0:
        raise ValueError("percentage must be > 0")

    n_select = max(1, int(round(len(files) * percentage)))
    n_select = min(n_select, len(files))

    rng = random.Random(seed)
    selected = rng.sample(files, n_select)

    return sorted(selected, key=lambda p: p.as_posix())


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)

    return h.hexdigest()


def _manifest_metadata(
    *,
    corpus_name: str,
    period: str,
    seed: int,
    percentage: float,
    n_files: int,
    include_sha256: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "manifest_type": "corpus_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_name": corpus_name,
        "period": period,
        "seed": seed,
        "percentage": percentage,
        "n_files": n_files,
        "include_sha256": include_sha256,
        "source": source,
    }


def write_corpus_manifest(
    records_or_files: Iterable[dict[str, Any] | str | Path],
    manifest_path: str | Path,
    *,
    corpus_name: str,
    period: str,
    seed: int,
    percentage: float,
    include_sha256: bool = False,
    source: str = "selected_files",
) -> None:
    """
    Write a JSONL manifest describing the exact files included in a corpus run.

    The function accepts either plain file paths or richer records. For corpus
    building, richer records are preferred because they can include extraction
    status, number of extracted characters, and whether a document was included
    in the raw corpus.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for i, item in enumerate(records_or_files):
        if isinstance(item, dict):
            record = dict(item)
            if "source_path" in record:
                path = Path(record["source_path"])
            else:
                path = None
        else:
            path = Path(item)
            record = {"source_path": path.as_posix()}

        if path is not None:
            record.setdefault("index", i)
            record.setdefault("source_path", path.as_posix())
            record.setdefault("filename", path.name)
            record.setdefault("size_bytes", path.stat().st_size if path.exists() else None)
            if include_sha256 and path.exists():
                record.setdefault("sha256", sha256_file(path))

        records.append(record)

    records = sorted(records, key=lambda r: str(r.get("source_path", "")))
    for i, record in enumerate(records):
        record["index"] = i

    metadata = _manifest_metadata(
        corpus_name=corpus_name,
        period=period,
        seed=seed,
        percentage=percentage,
        n_files=len(records),
        include_sha256=include_sha256,
        source=source,
    )

    with manifest_path.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"_metadata": metadata}, ensure_ascii=False) + "\n")
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_corpus_manifest(
    manifest_path: str | Path,
    *,
    only_included: bool = False,
) -> list[Path]:
    """
    Read file paths from a JSONL corpus manifest.

    Parameters
    ----------
    manifest_path:
        Path to JSONL manifest.
    only_included:
        If True and the manifest contains an `included_in_raw_corpus` field,
        only records where this field is True are returned.
    """
    manifest_path = Path(manifest_path)
    files: list[Path] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "_metadata" in record:
                continue
            if only_included and record.get("included_in_raw_corpus") is False:
                continue
            files.append(Path(record["source_path"]))

    return sorted(files, key=lambda p: p.as_posix())


def read_pdf_paths_from_selected_records(
    selected_records_jsonl: str | Path,
    *,
    pdf_dir: str | Path | None = None,
) -> list[Path]:
    """
    Read PDF paths from selected PDF records JSONL.

    The selection script usually writes records with `_pdf_path`. If `_pdf_path`
    is absent, this function falls back to `pdf_path`, `path`, or builds
    `pdf_dir / f"{paperId}.pdf"`.
    """
    selected_records_jsonl = Path(selected_records_jsonl)
    pdf_dir_path = Path(pdf_dir) if pdf_dir is not None else None

    paths: list[Path] = []

    with selected_records_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Bad JSON in selected PDF records at line {line_no}: {exc}")
                continue

            raw_path = (
                record.get("_pdf_path")
                or record.get("pdf_path")
                or record.get("path")
                or record.get("source_path")
            )

            if raw_path:
                path = Path(raw_path)
            else:
                paper_id = record.get("paperId") or record.get("paper_id")
                if not paper_id:
                    print(
                        f"[WARN] Selected PDF record line {line_no} has neither "
                        "_pdf_path/pdf_path/source_path nor paperId. Skipping."
                    )
                    continue
                if pdf_dir_path is None:
                    raise ValueError(
                        "Selected PDF records do not contain PDF paths. "
                        "Provide --input-dirs PDF_DIR so paths can be built from paperId."
                    )
                path = pdf_dir_path / f"{paper_id}.pdf"

            paths.append(path)

    # Deterministic order and duplicate removal.
    unique_paths = sorted({p for p in paths}, key=lambda p: p.as_posix())
    print(f"PDF paths loaded from selected records: {len(unique_paths)}")

    return unique_paths


def raw_corpus_item_to_text(item: Any) -> str:
    """
    Convert one raw-corpus item to text.

    Expected input is usually a string. This function also supports dicts with
    common text fields, which makes the merge more robust if XML corpus records
    were saved with metadata.
    """
    if item is None:
        return ""

    if isinstance(item, str):
        return item

    if isinstance(item, list):
        return " ".join(str(x) for x in item if x is not None)

    if isinstance(item, dict):
        for key in (
            "text",
            "raw_text",
            "cleaned_text",
            "body_text",
            "abstract",
            "title_abstract_text",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # Last-resort fallback: concatenate string values only.
        values = [v for v in item.values() if isinstance(v, str) and v.strip()]
        return "\n".join(values)

    return str(item)


def cleanup_xml_raw_text(text: str, *, remove_citations: bool = True) -> str:
    """
    Lightweight cleanup for XML-derived text.

    This delegates to text_normalization.cleanup_xml_raw_text(), which does NOT
    call fix_pdf_hyphenation() and removes only structured bracketed numeric
    citations by default, e.g. [4,5] or [75-78].
    """
    return textnorm.cleanup_xml_raw_text(text, remove_citations=remove_citations)


def cleanup_raw_text_for_merge(text: str, cleanup_mode: str) -> str:
    """
    Cleanup raw text during XML/PDF merge.

    cleanup_mode:
      - "none": no extra cleanup, only strip;
      - "xml": XML-safe cleanup with bracketed numeric citation removal;
      - "xml_no_citations": XML-safe cleanup but keep bracketed citations;
      - "pdf": full PDF cleanup, including conservative PDF hyphenation repair.

    In the current pipeline, PDF raw corpus is already cleaned when extracted,
    so merge usually uses "none" for the temporary PDF raw corpus.
    """
    if cleanup_mode == "none":
        return text.strip()
    if cleanup_mode == "xml":
        return cleanup_xml_raw_text(text, remove_citations=True)
    if cleanup_mode == "xml_no_citations":
        return cleanup_xml_raw_text(text, remove_citations=False)
    if cleanup_mode == "pdf":
        return textnorm.cleanup_pdf_extracted_text(text)
    raise ValueError(f"Unsupported cleanup_mode: {cleanup_mode}")


def merge_raw_corpora(
    input_corpus_specs: Iterable[tuple[str | Path, str]],
    output_corpus_path: str | Path,
) -> dict[str, Any]:
    """
    Stream-merge several JSON raw corpora into one JSON list of strings.

    This does not load the large XML corpus into memory. Each input corpus is
    expected to be a JSON list. Items can be plain strings or dict records with
    a text-like field.

    input_corpus_specs is a list of (path, cleanup_mode) pairs.
    """
    output_corpus_path = Path(output_corpus_path)
    output_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    input_specs = [(Path(p), mode) for p, mode in input_corpus_specs if p is not None]
    stats: dict[str, Any] = {
        "output_corpus_path": output_corpus_path.as_posix(),
        "inputs": [],
        "total_docs": 0,
        "total_chars": 0,
    }

    print("Merging raw corpora")
    print("-------------------")

    with output_corpus_path.open("w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for input_path, cleanup_mode in input_specs:
            input_docs = 0
            input_chars = 0
            print(f"Reading raw corpus: {input_path}")
            print(f"  cleanup mode: {cleanup_mode}")

            with input_path.open("r", encoding="utf-8") as f_in:
                for item in ijson.items(f_in, "item"):
                    text = raw_corpus_item_to_text(item)
                    text = cleanup_raw_text_for_merge(text, cleanup_mode)
                    text = text.strip()
                    if not text:
                        continue

                    if not first:
                        f_out.write(",\n")
                    json.dump(text, f_out, ensure_ascii=False)
                    first = False

                    input_docs += 1
                    input_chars += len(text)

                    if input_docs % 10_000 == 0:
                        print(f"  merged {input_docs} documents from {input_path.name}")

            stats["inputs"].append(
                {
                    "path": input_path.as_posix(),
                    "cleanup_mode": cleanup_mode,
                    "docs": input_docs,
                    "chars": input_chars,
                }
            )
            stats["total_docs"] += input_docs
            stats["total_chars"] += input_chars
            print(f"  done: {input_docs} docs, {input_chars} chars")

        f_out.write("\n]")

    print(f"Merged raw corpus saved to: {output_corpus_path}")
    print(f"Merged docs total: {stats['total_docs']}")

    return stats

# --- PDF/text utilities -------------------------------------------------------


def reset_eof_of_pdf_return_stream(pdf_stream_in: List[bytes]) -> List[bytes]:
    """
    Given a raw PDF byte stream, trim everything after the last %%EOF marker.

    This repair helper is retained for compatibility with earlier experiments,
    but normal extraction is now handled in pdf_extraction.py.
    """
    stream_len = len(pdf_stream_in)
    actual_line = stream_len
    for i, x in enumerate(pdf_stream_in[::-1]):
        if b"%%EOF" in x:
            actual_line = stream_len - i
            print(
                f"EOF found at line position {-i} = actual {actual_line}, "
                f"with value {x}"
            )
            break

    return pdf_stream_in[:actual_line]


def rewrite_pdf(filepath: str | Path) -> str:
    """
    Attempt to repair a PDF by removing bytes after the last '%%EOF' marker.
    """
    filepath = Path(filepath)
    with filepath.open("rb") as p:
        pdf_lines = p.readlines()

    trimmed_lines = reset_eof_of_pdf_return_stream(pdf_lines)
    new_filepath = filepath.with_name(filepath.stem + "_fixed.pdf")

    with new_filepath.open("wb") as f:
        f.writelines(trimmed_lines)

    return new_filepath.as_posix()


def iter_text_chunks(text: str, max_chars: int = 1_000_000) -> Iterable[str]:
    """
    Yield whitespace-aware chunks no longer than approximately max_chars.

    This prevents spaCy [E088] errors on very long XML/PDF-derived documents.
    We do not split by sentences because sentence parsing is disabled.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        yield text
        return

    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            # Prefer a whitespace boundary near the end of the chunk.
            split_at = text.rfind(" ", start, end)
            if split_at <= start + max_chars // 2:
                split_at = text.rfind("\n", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        start = max(end, start + 1)


def lemmatize_text(
    text: str,
    *,
    drop_numeric: bool = True,
    spacy_chunk_chars: int = 1_000_000,
) -> List[str]:
    """
    Lemmatize text using spaCy, remove English stopwords and punctuation.

    Long documents are processed in chunks to avoid spaCy [E088] max_length
    errors and excessive memory usage.

    Parameters
    ----------
    text:
        Cleaned document text.
    drop_numeric:
        If True, remove standalone numeric tokens.
    spacy_chunk_chars:
        Maximum character length passed to spaCy at once.
    """
    lemmas: List[str] = []

    for chunk in iter_text_chunks(text, max_chars=spacy_chunk_chars):
        doc = nlp(chunk)

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            if drop_numeric and token.like_num:
                continue

            word = token.text.lower()
            if word in stop_words:
                continue

            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue

            lemmas.append(lemma)

    return lemmas


# --- Corpus building ----------------------------------------------------------


def get_corpus(
    data_folders: List[str | Path],
    save_name: str | Path,
    percentage: float = 1.0,
    failed_log_path: str | Path = "failed_process_pdf.log",
    *,
    seed: int = DEFAULT_SEED,
    corpus_name: str = "corpus",
    period: str = "unknown_period",
    manifest_path: str | Path | None = None,
    corpus_manifest: str | Path | None = None,
    pdf_selected_records_jsonl: str | Path | None = None,
    include_sha256: bool = False,
    pdf_backend: str = "pymupdf_columns",
    pdf_fallback_backend: str | None = "pypdf",
    use_pdf_fallback: bool = True,
    suspicious_min_clean_chars: int = 1000,
) -> list[Path]:
    """
    Build a raw cleaned-text corpus from folders with PDF files.

    If `corpus_manifest` is provided, files are loaded from that manifest.
    Otherwise, files are collected from `data_folders`, sorted, and selected with
    deterministic sampling.

    PDF extraction is delegated to pdf_extraction.py. The text saved in the raw
    corpus is already cleaned with text_normalization.cleanup_extracted_text().
    """
    print("Creating a corpus")
    if corpus_manifest is not None:
        selected_files = read_corpus_manifest(corpus_manifest)
        selection_source = f"manifest:{corpus_manifest}"
    elif pdf_selected_records_jsonl is not None:
        pdf_dir = Path(data_folders[0]) if data_folders else None
        selected_files = read_pdf_paths_from_selected_records(
            pdf_selected_records_jsonl,
            pdf_dir=pdf_dir,
        )
        selected_files = select_random_files(selected_files, percentage=percentage, seed=seed)
        selection_source = f"selected_pdf_records:{pdf_selected_records_jsonl}"
    else:
        all_files = collect_files(data_folders, suffixes=DEFAULT_SUFFIXES)
        selected_files = select_random_files(all_files, percentage=percentage, seed=seed)
        selection_source = "deterministic_sampling"

    print(f"Selected PDF files: {len(selected_files)}")
    print(f"PDF backend: {pdf_backend}")
    if use_pdf_fallback and pdf_fallback_backend:
        print(f"PDF fallback backend: {pdf_fallback_backend}")

    corpus: List[str] = []
    manifest_records: list[dict[str, Any]] = []
    processed_docs = 0
    failed_log_path = Path(failed_log_path)
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    for file_path in selected_files:
        file_path = Path(file_path)
        record: dict[str, Any] = {
            "source_path": file_path.as_posix(),
            "filename": file_path.name,
            "size_bytes": file_path.stat().st_size if file_path.exists() else None,
            "included_in_raw_corpus": False,
            "status": "not_processed",
            "pdf_backend_requested": pdf_backend,
            "pdf_backend_used": None,
            "pdf_fallback_backend": pdf_fallback_backend if use_pdf_fallback else None,
            "pdf_fallback_used": False,
            "n_pages": 0,
            "n_raw_chars": 0,
            "n_clean_chars": 0,
            "citation_markers_before_cleanup": 0,
            "elapsed_seconds": 0.0,
        }

        result = extract_pdf_text(
            file_path,
            backend=pdf_backend,
            fallback_backend=pdf_fallback_backend,
            use_fallback=use_pdf_fallback,
            suspicious_min_clean_chars=suspicious_min_clean_chars,
            apply_cleanup=True,
        )

        record.update(
            {
                "status": result.status,
                "pdf_backend_used": result.backend_used,
                "pdf_fallback_used": result.fallback_used,
                "n_pages": result.n_pages,
                "n_raw_chars": result.raw_chars,
                "n_clean_chars": result.clean_chars,
                "citation_markers_before_cleanup": result.citation_markers_before_cleanup,
                "elapsed_seconds": round(result.elapsed_seconds, 6),
                "error": result.error,
            }
        )

        if result.status not in {"ok", "ok_with_warnings"}:
            with failed_log_path.open("a", encoding="utf-8") as log_f:
                print(result.status, file=log_f)
                print(result.error, file=log_f)
                print(file_path, file=log_f)
            manifest_records.append(record)
            continue

        if not result.cleaned_text.strip():
            record["status"] = "empty_after_cleaning"
            manifest_records.append(record)
            continue

        corpus.append(result.cleaned_text)
        processed_docs += 1

        record["included_in_raw_corpus"] = True
        manifest_records.append(record)

    print("Processed docs count:", processed_docs)

    save_name = Path(save_name)
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with save_name.open("w", encoding="utf-8") as f:
        json.dump(textnorm.clean_object(corpus), f, ensure_ascii=False)

    print(f"Corpus saved to: {save_name}")

    if manifest_path is not None:
        write_corpus_manifest(
            manifest_records,
            manifest_path=manifest_path,
            corpus_name=corpus_name,
            period=period,
            seed=seed,
            percentage=percentage,
            include_sha256=include_sha256,
            source=selection_source,
        )
        print(f"Corpus manifest saved to: {manifest_path}")

    return selected_files


def preprocess_corpus_line_by_line(
    corpus_path: str | Path,
    preprocess_corpus_path: str | Path,
    *,
    drop_numeric: bool = True,
    spacy_chunk_chars: int = 1_000_000,
    failed_preprocess_log_path: str | Path | None = None,
) -> None:
    """
    Preprocess a JSON corpus line-by-line and save the tokenized corpus.
    """
    print("Preprocessing corpus")
    print("---------------------------")

    corpus_path = Path(corpus_path)
    preprocess_corpus_path = Path(preprocess_corpus_path)
    preprocess_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with corpus_path.open("r", encoding="utf-8") as f_in, preprocess_corpus_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        docs = ijson.items(f_in, "item")
        f_out.write("[\n")

        first = True
        processed_docs = 0
        total_tokens = 0
        empty_docs = 0
        chunked_docs = 0
        max_doc_chars_seen = 0

        failed_log_f = None
        if failed_preprocess_log_path is not None:
            failed_preprocess_log_path = Path(failed_preprocess_log_path)
            failed_preprocess_log_path.parent.mkdir(parents=True, exist_ok=True)
            failed_log_f = failed_preprocess_log_path.open("w", encoding="utf-8")

        try:
            for i, doc in enumerate(docs, start=1):
                if i == 1 or i % 10_000 == 0:
                    print(f"Preprocessed {i} documents")

                if not isinstance(doc, str):
                    doc = str(doc)

                doc_len = len(doc)
                max_doc_chars_seen = max(max_doc_chars_seen, doc_len)
                if doc_len > spacy_chunk_chars:
                    chunked_docs += 1

                try:
                    tokens = lemmatize_text(
                        doc,
                        drop_numeric=drop_numeric,
                        spacy_chunk_chars=spacy_chunk_chars,
                    )
                except Exception as exc:
                    if failed_log_f is not None:
                        failed_log_f.write(
                            json.dumps(
                                {
                                    "doc_index": i,
                                    "doc_chars": doc_len,
                                    "error": repr(exc),
                                    "text_preview": doc[:500],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    # Keep document alignment by writing an empty token list.
                    tokens = []

                processed_docs += 1
                total_tokens += len(tokens)
                if not tokens:
                    empty_docs += 1

                if not first:
                    f_out.write(",\n")
                json.dump(tokens, f_out, ensure_ascii=False)
                first = False
        finally:
            if failed_log_f is not None:
                failed_log_f.close()

        f_out.write("\n]")

    print(f"Preprocessed documents total: {processed_docs}")
    print(f"Preprocessed tokens total: {total_tokens}")
    print(f"Empty tokenized documents: {empty_docs}")
    print(f"Chunked long documents: {chunked_docs}")
    print(f"Max document length seen, chars: {max_doc_chars_seen}")
    print(f"Preprocessed corpus saved to: {preprocess_corpus_path}")


# --- CLI / main ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, merge, normalize, and preprocess XML/PDF corpora into tokenized JSON."
    )

    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["path_to_folder_with_pdf_articles"],
        help="One or more folders with PDF articles.",
    )
    parser.add_argument(
        "--corpus-name",
        default="dental",
        help="Corpus name used in output filenames and manifests.",
    )
    parser.add_argument(
        "--period",
        default="unknown_period",
        help="Corpus period label, e.g. pre2018 or post2018.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="Fraction of collected PDF files to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic PDF file selection.",
    )
    parser.add_argument(
        "--corpus-manifest",
        default=None,
        help="Optional existing JSONL manifest. If provided, files are loaded from it.",
    )
    parser.add_argument(
        "--pdf-selected-records-jsonl",
        default=None,
        help=(
            "Optional JSONL produced by select_valid_pdf_articles. If provided, "
            "PDF paths are read from _pdf_path/pdf_path/source_path, or built "
            "from paperId and the first --input-dirs directory."
        ),
    )
    parser.add_argument(
        "--xml-raw-corpus",
        default=None,
        help=(
            "Optional existing XML raw corpus JSON, e.g. xml_raw_corpus_after_2018.json. "
            "If provided together with PDFs, XML and PDF raw corpora are merged before preprocessing."
        ),
    )
    parser.add_argument(
        "--no-clean-xml-raw-corpus",
        action="store_true",
        help=(
            "Do not apply XML-safe cleanup to the XML raw corpus during merge. "
            "Default: apply Unicode/control-character/bracket-citation/whitespace cleanup, "
            "but NOT PDF hyphenation repair."
        ),
    )
    parser.add_argument(
        "--no-clean-xml-citations",
        action="store_true",
        help=(
            "Keep bracketed numeric XML citations such as [4,5]. By default, "
            "XML-safe cleanup removes only these structured bracketed citations."
        ),
    )
    parser.add_argument(
        "--include-sha256",
        action="store_true",
        help="Include SHA256 hashes in the corpus manifest. Slower but more auditable.",
    )
    parser.add_argument(
        "--pdf-backend",
        choices=["pypdf", "pymupdf", "pymupdf_columns", "auto"],
        default="pymupdf_columns",
        help="PDF extraction backend. Canonical default: pymupdf_columns.",
    )
    parser.add_argument(
        "--pdf-fallback-backend",
        choices=["pypdf", "pymupdf", "pymupdf_columns"],
        default="pypdf",
        help="Fallback backend used when primary extraction is suspicious.",
    )
    parser.add_argument(
        "--no-pdf-fallback",
        action="store_true",
        help="Disable fallback PDF extraction backend.",
    )
    parser.add_argument(
        "--suspicious-min-clean-chars",
        type=int,
        default=1000,
        help="Try fallback when primary extraction returns fewer cleaned characters.",
    )
    parser.add_argument(
        "--keep-numeric-tokens",
        action="store_true",
        help="Keep standalone numeric tokens during preprocessing. Default: remove them.",
    )
    parser.add_argument(
        "--spacy-max-length",
        type=int,
        default=10_000_000,
        help=(
            "spaCy nlp.max_length safety limit in characters. Long documents are "
            "processed in chunks, so this rarely needs to be changed."
        ),
    )
    parser.add_argument(
        "--spacy-chunk-chars",
        type=int,
        default=1_000_000,
        help=(
            "Maximum number of characters passed to spaCy at once during preprocessing. "
            "Use this to avoid spaCy [E088] errors on very long documents."
        ),
    )
    parser.add_argument(
        "--reuse-raw-corpus",
        action="store_true",
        help=(
            "Skip XML/PDF extraction and merging if the raw corpus JSON already exists. "
            "Useful for restarting after a preprocessing crash."
        ),
    )
    parser.add_argument(
        "--reuse-preprocessed-corpus",
        action="store_true",
        help=(
            "Skip preprocessing if the preprocessed corpus JSON already exists."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for corpora, manifests, and logs.",
    )
    parser.add_argument(
        "--raw-corpus-path",
        default=None,
        help="Optional path for raw corpus JSON.",
    )
    parser.add_argument(
        "--preprocessed-corpus-path",
        default=None,
        help="Optional path for preprocessed corpus JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Extract and merge XML/PDF text sources, normalize the corpus,
    preprocess it, and save the tokenized *.preprocessed.json output.
    """
    args = parse_args()
    initialize_nlp_resources()
    require_ijson()

    # Make spaCy limit explicit and reproducible. Documents longer than
    # --spacy-chunk-chars are split before calling nlp(), so max_length is only
    # a safety limit for individual chunks.
    nlp.max_length = max(args.spacy_max_length, args.spacy_chunk_chars + 1_000)

    output_dir = Path(args.output_dir)
    run_name = f"{args.corpus_name}_{args.period}_seed{args.seed}"

    corpus_save_path = Path(args.raw_corpus_path) if args.raw_corpus_path else output_dir / "corpora" / f"{run_name}.raw.json"
    prep_corpus_save_path = (
        Path(args.preprocessed_corpus_path)
        if args.preprocessed_corpus_path
        else output_dir / "corpora" / f"{run_name}.preprocessed.json"
    )
    manifest_path = output_dir / "manifests" / f"{run_name}.manifest.jsonl"
    failed_log_path = output_dir / "logs" / f"{run_name}.failed_pdf.log"

    # 1. Build raw text corpus.
    #
    # Restart mode: if --reuse-raw-corpus is set and the raw corpus path already
    # exists, skip expensive XML/PDF extraction and merge.
    if args.reuse_raw_corpus and corpus_save_path.exists():
        print(f"Reusing existing raw corpus: {corpus_save_path}")
    else:
        #
        # Old mode: PDF-only corpus is written directly to corpus_save_path.
        # New mode: if --xml-raw-corpus is provided, we first build a temporary
        # PDF raw corpus and then stream-merge XML + PDF raw corpora into
        # corpus_save_path. This avoids loading the huge XML corpus into memory.
        if args.xml_raw_corpus:
            # XML raw text should not go through PDF-specific hyphenation repair.
            # Use XML-safe cleanup by default, or no extra cleanup if requested.
            if args.no_clean_xml_raw_corpus:
                xml_cleanup_mode = "none"
            elif args.no_clean_xml_citations:
                xml_cleanup_mode = "xml_no_citations"
            else:
                xml_cleanup_mode = "xml"
            raw_inputs: list[tuple[Path, str]] = [(Path(args.xml_raw_corpus), xml_cleanup_mode)]

            # Build PDF corpus only if PDF inputs were provided.
            if args.pdf_selected_records_jsonl or args.corpus_manifest or args.input_dirs:
                pdf_raw_corpus_path = output_dir / "corpora" / f"{run_name}.pdf.raw.json"
                get_corpus(
                    data_folders=args.input_dirs,
                    save_name=pdf_raw_corpus_path,
                    percentage=args.percentage,
                    failed_log_path=failed_log_path,
                    seed=args.seed,
                    corpus_name=args.corpus_name,
                    period=args.period,
                    manifest_path=manifest_path,
                    corpus_manifest=args.corpus_manifest,
                    pdf_selected_records_jsonl=args.pdf_selected_records_jsonl,
                    include_sha256=args.include_sha256,
                    pdf_backend=args.pdf_backend,
                    pdf_fallback_backend=args.pdf_fallback_backend,
                    use_pdf_fallback=not args.no_pdf_fallback,
                    suspicious_min_clean_chars=args.suspicious_min_clean_chars,
                )
                # The temporary PDF raw corpus is already cleaned during extraction,
                # so do not clean it again during merge.
                raw_inputs.append((pdf_raw_corpus_path, "none"))

            merge_raw_corpora(
                raw_inputs,
                corpus_save_path,
            )
        else:
            get_corpus(
                data_folders=args.input_dirs,
                save_name=corpus_save_path,
                percentage=args.percentage,
                failed_log_path=failed_log_path,
                seed=args.seed,
                corpus_name=args.corpus_name,
                period=args.period,
                manifest_path=manifest_path,
                corpus_manifest=args.corpus_manifest,
                pdf_selected_records_jsonl=args.pdf_selected_records_jsonl,
                include_sha256=args.include_sha256,
                pdf_backend=args.pdf_backend,
                pdf_fallback_backend=args.pdf_fallback_backend,
                use_pdf_fallback=not args.no_pdf_fallback,
                suspicious_min_clean_chars=args.suspicious_min_clean_chars,
            )

    # 2. Preprocess corpus: lemmatize, remove stopwords/punctuation.
    # Restart mode: if --reuse-preprocessed-corpus is set and the preprocessed
    # corpus exists, skip this step.
    failed_preprocess_log_path = output_dir / "logs" / f"{run_name}.failed_preprocess.jsonl"

    if args.reuse_preprocessed_corpus and prep_corpus_save_path.exists():
        print(f"Reusing existing preprocessed corpus: {prep_corpus_save_path}")
    else:
        preprocess_corpus_line_by_line(
            corpus_save_path,
            prep_corpus_save_path,
            drop_numeric=not args.keep_numeric_tokens,
            spacy_chunk_chars=args.spacy_chunk_chars,
            failed_preprocess_log_path=failed_preprocess_log_path,
        )


if __name__ == "__main__":
    main()
