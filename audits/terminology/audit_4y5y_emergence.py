#!/usr/bin/env python3
"""
audit_4y5y_emergence.py

Audit whether 4Y/5Y zirconia notation is genuinely post-2018-specific in the
ANALYZED CORPUS, rather than merely absent from the pre-2018 Word2Vec
vocabulary because of preprocessing, phrase detection, or min_count filtering.

The audit follows the actual representation chain:

    merged cleaned raw corpus
        -> preprocessed token corpus
        -> shared-phraser output corpus
        -> Word2Vec vocabulary

Default audited terms:
    4y_psz
    5y_psz
    4y_tzp
    5y_tzp

This script does not retrain anything and does not modify the corpus/models.

Strong support for a "post-2018-specific notation in the analyzed corpus"
claim requires, for a term:

    pre raw occurrences == 0
    pre preprocessed evidence == 0
    pre phrased canonical-token count == 0
    pre model absent

and:

    post raw occurrences > 0
    post preprocessed evidence > 0
    post phrased canonical-token count > 0
    post model present

If a term occurs in the pre-2018 raw/preprocessed corpus but is missing from
the pre-2018 model, the script flags the emergence claim as unsafe: the
absence can then be explained by phrase formation or Word2Vec min_count.

IMPORTANT LIMITATION
--------------------
This audits the literal 4Y/5Y TZP/PSZ NOTATION represented by the pipeline.
It does not establish that the underlying zirconia composition/concept itself
did not exist before 2018 under another wording (for example "4 mol% yttria").

Expected corpus formats
-----------------------
--pre-raw / --post-raw:
    JSON list of strings: the cleaned merged raw corpus actually passed to
    preprocessing.

--pre-preprocessed / --post-preprocessed:
    JSON list of token lists.

--pre-phrased / --post-phrased:
    JSON list of token lists after application of the shared Phraser.

Final vocabulary:
    supply either:
      --pre-vocab / --post-vocab
    where each CSV contains token,count,
    OR:
      --pre-model / --post-model
    pointing to gensim Word2Vec models.

Outputs
-------
emergence_audit_summary.csv
stage_counts_long.csv
raw_variant_counts.csv  (strict / ambiguous_spaced / excluded_decimal_prefix)
raw_occurrence_examples.csv  (same evidence classes)
claim_assessment.txt
audit_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator, Any


DEFAULT_TERMS = ("4y_psz", "5y_psz", "4y_tzp", "5y_tzp")

DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
})

SUBSCRIPT_SUPERSCRIPT_TRANSLATION = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
})


# ---------------------------------------------------------------------------
# Streaming JSON-array reader using only the standard library
# ---------------------------------------------------------------------------

def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """
    Stream items from a top-level JSON array without loading the whole file.

    Works for the large *.raw.json, *.preprocessed.json and
    *.shared_phrased.json files produced by this project.
    """
    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        finished = False
        eof = False

        while not finished:
            # Refill when little/no unread data remains.
            if pos >= len(buf) and not eof:
                buf = f.read(chunk_size)
                pos = 0
                if buf == "":
                    eof = True

            if not started:
                # Find top-level '['.
                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if pos < len(buf):
                        if buf[pos] != "[":
                            raise ValueError(
                                f"{path}: expected top-level JSON array '['"
                            )
                        pos += 1
                        started = True
                        break
                    if eof:
                        raise ValueError(f"{path}: empty/incomplete JSON")
                    more = f.read(chunk_size)
                    if more == "":
                        eof = True
                    buf = buf[pos:] + more
                    pos = 0

            while started and not finished:
                # Skip whitespace and commas.
                while True:
                    while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                        pos += 1

                    if pos < len(buf):
                        break

                    if eof:
                        raise ValueError(f"{path}: unexpected EOF inside JSON array")

                    more = f.read(chunk_size)
                    if more == "":
                        eof = True
                    buf = buf[pos:] + more
                    pos = 0

                if buf[pos] == "]":
                    finished = True
                    pos += 1
                    break

                # Decode one item. If incomplete, append more input.
                while True:
                    try:
                        item, end = decoder.raw_decode(buf, pos)
                        pos = end
                        yield item

                        # Compact buffer periodically.
                        if pos > 4 * chunk_size:
                            buf = buf[pos:]
                            pos = 0
                        break

                    except json.JSONDecodeError as e:
                        if eof:
                            raise ValueError(
                                f"{path}: invalid/incomplete JSON near char "
                                f"{e.pos}: {e.msg}"
                            ) from e

                        # Preserve unread portion and append another chunk.
                        buf = buf[pos:]
                        pos = 0
                        more = f.read(chunk_size)
                        if more == "":
                            eof = True
                        buf += more


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_term(term: str) -> tuple[str, str]:
    """4y_psz -> ('4y', 'psz')."""
    t = term.strip().lower()
    m = re.fullmatch(r"([345]y)_(psz|tzp)", t)
    if not m:
        raise ValueError(
            f"Unsupported term {term!r}. Expected notation such as 4y_psz."
        )
    return m.group(1), m.group(2)


def normalize_text_for_raw_matching(text: str) -> str:
    """
    Conservative Unicode normalization used ONLY for matching/reporting.
    It does not alter any source corpus file.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(DASH_TRANSLATION)
    text = text.translate(SUBSCRIPT_SUPERSCRIPT_TRANSLATION)
    return text


def raw_patterns(term: str) -> dict[str, re.Pattern[str]]:
    """
    Return three deliberately separate raw-text matchers.

    strict
        Accepted notation evidence. The concentration digit must be directly
        attached to Y, e.g.:
            4Y-TZP
            4Y - TZP
            4Y TZP
            4YTZP

        This matcher explicitly DOES NOT match:
            4 Y-TZP          (can be a table/list boundary artifact)
            2.5Y-TZP         (must not be misread as 5Y-TZP)
            4.5Y-TZP         (must not be misread as 5Y-TZP)

    ambiguous_spaced
        Candidate forms with whitespace between the digit and Y, e.g.
        "4 Y-TZP". These are reported for manual review but are NOT counted
        as strict evidence for the emergence claim.

    excluded_decimal_prefix
        Decimal compositions such as 2.5Y-TZP or 4.5Y-TZP that would have
        produced false 5Y-TZP hits with the earlier matcher. These are
        reported explicitly as excluded candidates and never count as
        evidence for the target term.
    """
    left, right = parse_term(term)
    digit = re.escape(left[0])
    right_rx = re.escape(right)

    # Do not start inside an alphanumeric token or immediately after '.', ','
    # so that "2.5Y-TZP" cannot become a false "5Y-TZP".
    left_boundary = r"(?<![A-Za-z0-9_.,])"
    right_boundary = r"(?![A-Za-z0-9_])"

    strict = re.compile(
        rf"{left_boundary}"
        rf"{digit}[Yy]\s*(?:-\s*)?{right_rx}"
        rf"{right_boundary}",
        flags=re.IGNORECASE,
    )

    ambiguous_spaced = re.compile(
        rf"{left_boundary}"
        rf"{digit}\s+[Yy]\s*(?:-\s*)?{right_rx}"
        rf"{right_boundary}",
        flags=re.IGNORECASE,
    )

    # Specifically capture decimal concentrations whose fractional digit equals
    # the audited digit. Example for term=5y_tzp: 2.5Y-TZP, 4.5Y-TZP.
    excluded_decimal_prefix = re.compile(
        rf"(?<![A-Za-z0-9_])"
        rf"\d+\s*[\.,]\s*{digit}[Yy]\s*(?:-\s*)?{right_rx}"
        rf"{right_boundary}",
        flags=re.IGNORECASE,
    )

    return {
        "strict": strict,
        "ambiguous_spaced": ambiguous_spaced,
        "excluded_decimal_prefix": excluded_decimal_prefix,
    }


def to_tokens(item: Any) -> list[str]:
    if isinstance(item, list):
        return [str(x).lower() for x in item]
    if isinstance(item, str):
        return item.lower().split()
    return [str(item).lower()]


def file_meta(path: Path, do_sha256: bool = False) -> dict:
    st = path.stat()
    out = {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

    if do_sha256:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        out["sha256"] = h.hexdigest()

    return out


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Raw-corpus audit
# ---------------------------------------------------------------------------

def make_snippet(text: str, start: int, end: int, context: int) -> str:
    lo = max(0, start - context)
    hi = min(len(text), end + context)
    snippet = re.sub(r"\s+", " ", text[lo:hi]).strip()

    if lo > 0:
        snippet = "..." + snippet
    if hi < len(text):
        snippet += "..."

    return snippet


def audit_raw_corpus(
    path: Path,
    period: str,
    terms: list[str],
    max_examples: int,
    snippet_chars: int,
) -> tuple[dict[str, dict], list[dict], list[dict], dict]:

    patterns = {term: raw_patterns(term) for term in terms}

    stats = {
        term: {
            "strict_occurrences": 0,
            "strict_documents": 0,
            "ambiguous_spaced_occurrences": 0,
            "ambiguous_spaced_documents": 0,
            "excluded_decimal_prefix_occurrences": 0,
            "excluded_decimal_prefix_documents": 0,
        }
        for term in terms
    }

    variant_counters = {
        term: {
            "strict": Counter(),
            "ambiguous_spaced": Counter(),
            "excluded_decimal_prefix": Counter(),
        }
        for term in terms
    }
    example_counts = Counter()
    examples: list[dict] = []

    n_docs = 0
    non_string_items = 0
    total_chars = 0

    for doc_idx, item in enumerate(iter_json_array(path)):
        n_docs += 1

        if isinstance(item, str):
            text = item
        elif isinstance(item, list):
            text = " ".join(str(x) for x in item)
            non_string_items += 1
        else:
            text = str(item)
            non_string_items += 1

        total_chars += len(text)
        normalized = normalize_text_for_raw_matching(text)

        for term, class_patterns in patterns.items():
            for evidence_class, pattern in class_patterns.items():
                matches = list(pattern.finditer(normalized))
                if not matches:
                    continue

                if evidence_class == "strict":
                    stats[term]["strict_documents"] += 1
                    stats[term]["strict_occurrences"] += len(matches)
                elif evidence_class == "ambiguous_spaced":
                    stats[term]["ambiguous_spaced_documents"] += 1
                    stats[term]["ambiguous_spaced_occurrences"] += len(matches)
                elif evidence_class == "excluded_decimal_prefix":
                    stats[term]["excluded_decimal_prefix_documents"] += 1
                    stats[term]["excluded_decimal_prefix_occurrences"] += len(matches)

                for match in matches:
                    surface = match.group(0)
                    variant_counters[term][evidence_class][surface] += 1

                    key = (period, term, evidence_class)
                    if example_counts[key] < max_examples:
                        examples.append({
                            "period": period,
                            "term": term,
                            "evidence_class": evidence_class,
                            "document_index": doc_idx,
                            "matched_surface": surface,
                            "snippet": make_snippet(
                                normalized,
                                match.start(),
                                match.end(),
                                snippet_chars,
                            ),
                        })
                        example_counts[key] += 1

    variant_rows = []
    for term in terms:
        for evidence_class in (
            "strict",
            "ambiguous_spaced",
            "excluded_decimal_prefix",
        ):
            for surface, count in variant_counters[term][evidence_class].most_common():
                variant_rows.append({
                    "period": period,
                    "term": term,
                    "evidence_class": evidence_class,
                    "surface_variant": surface,
                    "count": count,
                })

    corpus_stats = {
        "n_documents": n_docs,
        "non_string_items": non_string_items,
        "total_characters": total_chars,
    }

    return stats, variant_rows, examples, corpus_stats


# ---------------------------------------------------------------------------
# Token-corpus audit
# ---------------------------------------------------------------------------

def count_term_evidence_in_tokens(tokens: list[str], term: str) -> dict:
    """
    For 4y_psz distinguish:
      * canonical phrase token: 4y_psz
      * adjacent components: 4y, psz
      * hyphenated token: 4y-psz
      * concatenated token: 4ypsz
    """
    left, right = parse_term(term)
    canonical = term
    hyphen = f"{left}-{right}"
    concat = f"{left}{right}"

    canonical_count = sum(tok == canonical for tok in tokens)
    hyphen_count = sum(tok == hyphen for tok in tokens)
    concat_count = sum(tok == concat for tok in tokens)
    pair_count = sum(
        a == left and b == right
        for a, b in zip(tokens, tokens[1:])
    )

    return {
        "canonical_token_count": canonical_count,
        "adjacent_pair_count": pair_count,
        "hyphen_token_count": hyphen_count,
        "concat_token_count": concat_count,
        "evidence_count": (
            canonical_count + pair_count + hyphen_count + concat_count
        ),
    }


def audit_token_corpus(
    path: Path,
    terms: list[str],
) -> tuple[dict[str, dict], dict]:

    stats = {}

    for term in terms:
        stats[term] = {
            "canonical_token_count": 0,
            "adjacent_pair_count": 0,
            "hyphen_token_count": 0,
            "concat_token_count": 0,
            "evidence_count": 0,
            "documents_with_evidence": 0,
            "left_component_count": 0,
            "right_component_count": 0,
        }

    n_docs = 0
    total_tokens = 0
    non_list_items = 0

    for item in iter_json_array(path):
        n_docs += 1

        if not isinstance(item, list):
            non_list_items += 1

        tokens = to_tokens(item)
        total_tokens += len(tokens)
        token_counter = Counter(tokens)

        for term in terms:
            left, right = parse_term(term)
            evidence = count_term_evidence_in_tokens(tokens, term)

            for metric in (
                "canonical_token_count",
                "adjacent_pair_count",
                "hyphen_token_count",
                "concat_token_count",
                "evidence_count",
            ):
                stats[term][metric] += evidence[metric]

            stats[term]["left_component_count"] += token_counter[left]
            stats[term]["right_component_count"] += token_counter[right]

            if evidence["evidence_count"] > 0:
                stats[term]["documents_with_evidence"] += 1

    corpus_stats = {
        "n_documents": n_docs,
        "total_tokens": total_tokens,
        "non_list_items": non_list_items,
    }

    return stats, corpus_stats


# ---------------------------------------------------------------------------
# Final vocabulary/model audit
# ---------------------------------------------------------------------------

def load_vocab_csv(path: Path) -> dict[str, int]:
    """
    Accept a saved vocabulary CSV with token/count, word/count, or term/count.
    """
    out: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"Empty/invalid vocabulary CSV: {path}")

        lower = {x.lower(): x for x in reader.fieldnames}
        token_col = (
            lower.get("token")
            or lower.get("word")
            or lower.get("term")
        )
        count_col = (
            lower.get("count")
            or lower.get("frequency")
            or lower.get("freq")
        )

        if not token_col or not count_col:
            raise ValueError(
                f"{path}: expected token/count columns; "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            token = (row.get(token_col) or "").strip().lower()
            if not token:
                continue

            try:
                count = int(float(row.get(count_col) or 0))
            except ValueError:
                count = 0

            out[token] = count

    return out


def load_model_vocab(path: Path) -> dict[str, int]:
    try:
        from gensim.models import Word2Vec, KeyedVectors
    except ImportError as exc:
        raise SystemExit(
            "Reading --pre-model/--post-model requires gensim. "
            "Alternatively pass the saved *.vocab.csv files."
        ) from exc

    try:
        model = Word2Vec.load(str(path))
        wv = model.wv
    except Exception:
        wv = KeyedVectors.load(str(path))

    return {
        token.lower(): int(wv.get_vecattr(token, "count"))
        for token in wv.index_to_key
    }


def get_vocab(
    vocab_path: Path | None,
    model_path: Path | None,
) -> tuple[dict[str, int], str]:

    if vocab_path is not None:
        return load_vocab_csv(vocab_path), f"vocab_csv:{vocab_path}"

    if model_path is not None:
        return load_model_vocab(model_path), f"model:{model_path}"

    raise ValueError(
        "Supply either vocabulary CSV or model path for each period."
    )


# ---------------------------------------------------------------------------
# Conservative claim assessment
# ---------------------------------------------------------------------------

def assess_term(
    term: str,
    pre_raw: dict,
    post_raw: dict,
    pre_pre: dict,
    post_pre: dict,
    pre_phr: dict,
    post_phr: dict,
    pre_vocab_count: int,
    post_vocab_count: int,
    min_count: int,
) -> tuple[str, str]:

    pre_model = pre_vocab_count > 0
    post_model = post_vocab_count > 0

    pre_raw_strict = int(pre_raw["strict_occurrences"])
    post_raw_strict = int(post_raw["strict_occurrences"])

    pre_raw_spaced = int(pre_raw["ambiguous_spaced_occurrences"])
    pre_raw_decimal_excluded = int(pre_raw["excluded_decimal_prefix_occurrences"])

    pre_pre_n = int(pre_pre["evidence_count"])
    post_pre_n = int(post_pre["evidence_count"])

    pre_phrase_n = int(pre_phr["canonical_token_count"])
    post_phrase_n = int(post_phr["canonical_token_count"])

    # First establish whether it is actually post-only in model vocabulary.
    if pre_model and post_model:
        return (
            "NOT_POST_ONLY_MODEL_VOCAB",
            "The canonical token is present in both Word2Vec vocabularies; "
            "do not describe it as post-2018-only.",
        )

    if not post_model:
        return (
            "NO_POST_MODEL_SUPPORT",
            "The canonical token is absent from the post-2018 Word2Vec "
            "vocabulary; the model-level emergence claim is unsupported.",
        )

    # From here: the canonical model token is post-only.
    #
    # Any strict pre-2018 raw evidence is enough to reject a literal
    # post-2018-emergence claim.
    if pre_raw_strict > 0:
        explanation = (
            f"Strict pre-2018 raw-corpus notation evidence exists "
            f"({pre_raw_strict} occurrence(s)), although the canonical model "
            "token is absent. Therefore model absence cannot establish that "
            "the notation first appeared after 2018."
        )

        if pre_phrase_n > 0 and pre_phrase_n < min_count:
            explanation += (
                f" The pre-2018 phrased count ({pre_phrase_n}) is below "
                f"min_count={min_count}, so Word2Vec vocabulary filtering "
                "can explain the model absence."
            )
        elif pre_pre_n > 0 and pre_phrase_n == 0:
            explanation += (
                " Preprocessed evidence exists but no canonical phrase token "
                "was formed, so phrase detection can explain the model absence."
            )

        if pre_raw_decimal_excluded > 0:
            explanation += (
                f" In addition, {pre_raw_decimal_excluded} decimal-prefixed "
                "candidate(s) such as 2.5Y/4.5Y notation were explicitly "
                "excluded from the target count."
            )

        return "UNSAFE_PRE2018_STRICT_RAW_EVIDENCE", explanation

    if pre_pre_n > 0:
        explanation = (
            "No strict pre-2018 raw-text hit was found, but pre-2018 "
            "preprocessed token evidence exists. Therefore the notation/token "
            "was represented before 2018 and should not be described as "
            "post-2018-only."
        )

        if pre_phrase_n > 0 and pre_phrase_n < min_count:
            explanation += (
                f" The pre-2018 phrased count ({pre_phrase_n}) is below "
                f"min_count={min_count}, so Word2Vec vocabulary filtering "
                "can explain its absence from the model."
            )

        if pre_raw_decimal_excluded > 0:
            explanation += (
                f" The raw-text audit also excluded "
                f"{pre_raw_decimal_excluded} decimal-prefixed false-candidate "
                "match(es) from the target count."
            )

        return "UNSAFE_PRE2018_PREPROCESSED_EVIDENCE", explanation

    if pre_phrase_n > 0:
        return (
            "UNSAFE_PRE2018_PHRASED_EVIDENCE",
            "The canonical phrase token occurs in the pre-2018 phrased corpus "
            "but is absent from the pre-2018 model. This is compatible with "
            "Word2Vec vocabulary filtering rather than emergence.",
        )

    # Check that the post side is coherent across stages.
    if post_raw_strict <= 0:
        return (
            "AMBIGUOUS_NO_POST_STRICT_RAW_EVIDENCE",
            "The canonical token is post-only in the model, but no strict "
            "post-2018 raw-corpus notation evidence was found. Check that the "
            "audited files belong to the same production run.",
        )

    if post_pre_n <= 0:
        return (
            "AMBIGUOUS_POST_PREPROCESSING_GAP",
            "Post-2018 strict raw notation exists but no corresponding "
            "preprocessed token evidence was found. Check normalization and "
            "tokenization.",
        )

    if post_phrase_n <= 0:
        return (
            "AMBIGUOUS_POST_PHRASE_GAP",
            "Post-2018 raw/preprocessed evidence exists, but the canonical "
            "phrase token is absent from the phrased corpus despite model "
            "presence. Check run provenance.",
        )

    if post_vocab_count < min_count:
        return (
            "AMBIGUOUS_MODEL_COUNT_BELOW_MIN_COUNT",
            f"The model reports count={post_vocab_count}, below the stated "
            f"min_count={min_count}; check model/config provenance.",
        )

    if pre_raw_spaced > 0:
        return (
            "SUPPORTED_WITH_AMBIGUOUS_SPACED_RAW_HITS",
            f"No strict pre-2018 notation evidence was found at the raw, "
            f"preprocessed, phrased, or model-vocabulary stages. However, "
            f"{pre_raw_spaced} spaced raw candidate(s) (e.g. '4 Y-TZP') were "
            "found and are reported separately for manual review because such "
            "forms can arise at table/list boundaries. They are not counted "
            "as strict notation evidence. Post-2018 strict notation is present "
            "through all pipeline stages."
        )

    explanation = (
        "The strict literal notation is absent from the pre-2018 raw, "
        "preprocessed, phrased, and model-vocabulary stages, while it is "
        "present through all corresponding post-2018 stages. This supports "
        "describing the notation/token as post-2018-specific in the analyzed "
        "corpus. It does not prove that the underlying zirconia composition "
        "or concept itself first appeared after 2018."
    )

    if pre_raw_decimal_excluded > 0:
        explanation += (
            f" The raw-text matcher separately identified and excluded "
            f"{pre_raw_decimal_excluded} decimal-prefixed form(s) (for example "
            "2.5Y/4.5Y notation) that are not instances of the audited term."
        )

    return "SUPPORTED_POST2018_SPECIFIC_NOTATION", explanation


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit 4Y/5Y TZP/PSZ emergence across the exact corpus pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--pre-raw", type=Path, required=True)
    p.add_argument("--post-raw", type=Path, required=True)

    p.add_argument("--pre-preprocessed", type=Path, required=True)
    p.add_argument("--post-preprocessed", type=Path, required=True)

    p.add_argument("--pre-phrased", type=Path, required=True)
    p.add_argument("--post-phrased", type=Path, required=True)

    pre_final = p.add_mutually_exclusive_group(required=True)
    pre_final.add_argument("--pre-vocab", type=Path)
    pre_final.add_argument("--pre-model", type=Path)

    post_final = p.add_mutually_exclusive_group(required=True)
    post_final.add_argument("--post-vocab", type=Path)
    post_final.add_argument("--post-model", type=Path)

    p.add_argument(
        "--terms",
        nargs="+",
        default=list(DEFAULT_TERMS),
        help="Canonical phrase tokens to audit.",
    )

    p.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Word2Vec min_count used for the final production models.",
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_4y5y_emergence_audit"),
    )

    p.add_argument(
        "--max-examples-per-term-period",
        type=int,
        default=5,
    )

    p.add_argument(
        "--snippet-chars",
        type=int,
        default=140,
    )

    p.add_argument(
        "--sha256-inputs",
        action="store_true",
        help="Hash all large inputs for provenance; slower.",
    )

    return p.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    paths = [
        args.pre_raw,
        args.post_raw,
        args.pre_preprocessed,
        args.post_preprocessed,
        args.pre_phrased,
        args.post_phrased,
        args.pre_vocab,
        args.post_vocab,
        args.pre_model,
        args.post_model,
    ]

    for path in paths:
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    for term in args.terms:
        parse_term(term)


def main() -> int:
    args = parse_args()
    validate_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    terms = [t.lower() for t in args.terms]

    print("1/4 Auditing cleaned raw corpora...")
    pre_raw, pre_variants, pre_examples, pre_raw_stats = audit_raw_corpus(
        args.pre_raw,
        "pre2018",
        terms,
        args.max_examples_per_term_period,
        args.snippet_chars,
    )
    post_raw, post_variants, post_examples, post_raw_stats = audit_raw_corpus(
        args.post_raw,
        "post2018",
        terms,
        args.max_examples_per_term_period,
        args.snippet_chars,
    )

    print("2/4 Auditing preprocessed token corpora...")
    pre_pre, pre_pre_stats = audit_token_corpus(
        args.pre_preprocessed,
        terms,
    )
    post_pre, post_pre_stats = audit_token_corpus(
        args.post_preprocessed,
        terms,
    )

    print("3/4 Auditing shared-phraser output corpora...")
    pre_phr, pre_phr_stats = audit_token_corpus(
        args.pre_phrased,
        terms,
    )
    post_phr, post_phr_stats = audit_token_corpus(
        args.post_phrased,
        terms,
    )

    print("4/4 Loading final Word2Vec vocabulary...")
    pre_vocab, pre_vocab_source = get_vocab(
        args.pre_vocab,
        args.pre_model,
    )
    post_vocab, post_vocab_source = get_vocab(
        args.post_vocab,
        args.post_model,
    )

    summary_rows: list[dict] = []
    long_rows: list[dict] = []
    assessment_lines: list[str] = []

    for term in terms:
        pre_model_count = int(pre_vocab.get(term, 0))
        post_model_count = int(post_vocab.get(term, 0))

        assessment, explanation = assess_term(
            term=term,
            pre_raw=pre_raw[term],
            post_raw=post_raw[term],
            pre_pre=pre_pre[term],
            post_pre=post_pre[term],
            pre_phr=pre_phr[term],
            post_phr=post_phr[term],
            pre_vocab_count=pre_model_count,
            post_vocab_count=post_model_count,
            min_count=args.min_count,
        )

        row = {
            "term": term,

            "pre_raw_strict_occurrences":
                pre_raw[term]["strict_occurrences"],
            "post_raw_strict_occurrences":
                post_raw[term]["strict_occurrences"],
            "pre_raw_strict_documents":
                pre_raw[term]["strict_documents"],
            "post_raw_strict_documents":
                post_raw[term]["strict_documents"],

            "pre_raw_ambiguous_spaced_occurrences":
                pre_raw[term]["ambiguous_spaced_occurrences"],
            "post_raw_ambiguous_spaced_occurrences":
                post_raw[term]["ambiguous_spaced_occurrences"],

            "pre_raw_excluded_decimal_prefix_occurrences":
                pre_raw[term]["excluded_decimal_prefix_occurrences"],
            "post_raw_excluded_decimal_prefix_occurrences":
                post_raw[term]["excluded_decimal_prefix_occurrences"],

            "pre_preprocessed_evidence": pre_pre[term]["evidence_count"],
            "post_preprocessed_evidence": post_pre[term]["evidence_count"],
            "pre_preprocessed_adjacent_pair":
                pre_pre[term]["adjacent_pair_count"],
            "post_preprocessed_adjacent_pair":
                post_pre[term]["adjacent_pair_count"],

            "pre_phrased_canonical_token":
                pre_phr[term]["canonical_token_count"],
            "post_phrased_canonical_token":
                post_phr[term]["canonical_token_count"],
            "pre_phrased_residual_pair":
                pre_phr[term]["adjacent_pair_count"],
            "post_phrased_residual_pair":
                post_phr[term]["adjacent_pair_count"],

            "pre_model_present": pre_model_count > 0,
            "post_model_present": post_model_count > 0,
            "pre_model_count": pre_model_count,
            "post_model_count": post_model_count,

            "claim_assessment": assessment,
            "assessment_explanation": explanation,
        }

        summary_rows.append(row)

        def add(period: str, stage: str, metric: str, value: Any) -> None:
            long_rows.append({
                "term": term,
                "period": period,
                "stage": stage,
                "metric": metric,
                "value": value,
            })

        for period, data in (
            ("pre2018", pre_raw[term]),
            ("post2018", post_raw[term]),
        ):
            for metric in (
                "strict_occurrences",
                "strict_documents",
                "ambiguous_spaced_occurrences",
                "ambiguous_spaced_documents",
                "excluded_decimal_prefix_occurrences",
                "excluded_decimal_prefix_documents",
            ):
                add(period, "raw", metric, data[metric])

        for period, data in (
            ("pre2018", pre_pre[term]),
            ("post2018", post_pre[term]),
        ):
            for metric in (
                "evidence_count",
                "canonical_token_count",
                "adjacent_pair_count",
                "hyphen_token_count",
                "concat_token_count",
                "documents_with_evidence",
                "left_component_count",
                "right_component_count",
            ):
                add(period, "preprocessed", metric, data[metric])

        for period, data in (
            ("pre2018", pre_phr[term]),
            ("post2018", post_phr[term]),
        ):
            for metric in (
                "evidence_count",
                "canonical_token_count",
                "adjacent_pair_count",
                "hyphen_token_count",
                "concat_token_count",
                "documents_with_evidence",
                "left_component_count",
                "right_component_count",
            ):
                add(period, "phrased", metric, data[metric])

        add(
            "pre2018",
            "model_vocab",
            "present",
            int(pre_model_count > 0),
        )
        add(
            "post2018",
            "model_vocab",
            "present",
            int(post_model_count > 0),
        )
        add("pre2018", "model_vocab", "count", pre_model_count)
        add("post2018", "model_vocab", "count", post_model_count)

        assessment_lines.append(f"{term}: {assessment}")
        assessment_lines.append(f"  {explanation}")
        assessment_lines.append("")

    write_csv(
        args.out_dir / "emergence_audit_summary.csv",
        summary_rows,
    )

    write_csv(
        args.out_dir / "stage_counts_long.csv",
        long_rows,
        ["term", "period", "stage", "metric", "value"],
    )

    write_csv(
        args.out_dir / "raw_variant_counts.csv",
        pre_variants + post_variants,
        ["period", "term", "evidence_class", "surface_variant", "count"],
    )

    write_csv(
        args.out_dir / "raw_occurrence_examples.csv",
        pre_examples + post_examples,
        [
            "period",
            "term",
            "evidence_class",
            "document_index",
            "matched_surface",
            "snippet",
        ],
    )

    assessment_header = (
        "4Y/5Y emergence audit\n"
        "======================\n\n"
        "Interpretation is deliberately conservative. A SUPPORTED result "
        "refers to literal notation/token emergence in the analyzed corpus, "
        "not to first discovery of the underlying material concept.\n\n"
    )

    (args.out_dir / "claim_assessment.txt").write_text(
        assessment_header + "\n".join(assessment_lines),
        encoding="utf-8",
    )

    input_paths = {
        "pre_raw": args.pre_raw,
        "post_raw": args.post_raw,
        "pre_preprocessed": args.pre_preprocessed,
        "post_preprocessed": args.post_preprocessed,
        "pre_phrased": args.pre_phrased,
        "post_phrased": args.post_phrased,
    }

    if args.pre_vocab:
        input_paths["pre_vocab"] = args.pre_vocab
    if args.post_vocab:
        input_paths["post_vocab"] = args.post_vocab
    if args.pre_model:
        input_paths["pre_model"] = args.pre_model
    if args.post_model:
        input_paths["post_model"] = args.post_model

    audit_json = {
        "terms": terms,
        "min_count": args.min_count,
        "vocabulary_sources": {
            "pre2018": pre_vocab_source,
            "post2018": post_vocab_source,
        },
        "input_files": {
            key: file_meta(path, do_sha256=args.sha256_inputs)
            for key, path in input_paths.items()
        },
        "corpus_stage_stats": {
            "pre2018_raw": pre_raw_stats,
            "post2018_raw": post_raw_stats,
            "pre2018_preprocessed": pre_pre_stats,
            "post2018_preprocessed": post_pre_stats,
            "pre2018_phrased": pre_phr_stats,
            "post2018_phrased": post_phr_stats,
        },
        "assessment": {
            row["term"]: {
                "status": row["claim_assessment"],
                "explanation": row["assessment_explanation"],
            }
            for row in summary_rows
        },
        "raw_matching_policy": {
            "strict": (
                "Digit directly attached to Y (e.g. 4Y-TZP, 4Y TZP, 4YTZP); "
                "counts as notation evidence."
            ),
            "ambiguous_spaced": (
                "Whitespace between digit and Y (e.g. 4 Y-TZP); reported "
                "separately for manual review and not counted as strict evidence."
            ),
            "excluded_decimal_prefix": (
                "Decimal compositions such as 2.5Y-TZP or 4.5Y-TZP are "
                "explicitly excluded from 5Y-TZP counts."
            ),
        },
        "limitations": [
            (
                "The audit establishes presence/absence of literal 4Y/5Y "
                "TZP/PSZ notation in the analyzed pipeline representations."
            ),
            (
                "It does not establish first discovery or first use of the "
                "underlying zirconia composition/concept under all possible "
                "alternative wording."
            ),
            (
                "The raw stage is the cleaned/extracted merged corpus actually "
                "passed to preprocessing, not an independent re-extraction of "
                "every source article."
            ),
        ],
    }

    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(audit_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 4Y/5Y EMERGENCE AUDIT ===")

    for row in summary_rows:
        print(
            f"{row['term']:8s} | "
            f"raw(strict) pre/post "
            f"{row['pre_raw_strict_occurrences']}/"
            f"{row['post_raw_strict_occurrences']} | "
            f"raw(spaced) "
            f"{row['pre_raw_ambiguous_spaced_occurrences']}/"
            f"{row['post_raw_ambiguous_spaced_occurrences']} | "
            f"raw(decimal-excluded) "
            f"{row['pre_raw_excluded_decimal_prefix_occurrences']}/"
            f"{row['post_raw_excluded_decimal_prefix_occurrences']} | "
            f"preproc "
            f"{row['pre_preprocessed_evidence']}/"
            f"{row['post_preprocessed_evidence']} | "
            f"phrase "
            f"{row['pre_phrased_canonical_token']}/"
            f"{row['post_phrased_canonical_token']} | "
            f"model "
            f"{row['pre_model_count']}/"
            f"{row['post_model_count']} | "
            f"{row['claim_assessment']}"
        )

    print(f"\nOutputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
