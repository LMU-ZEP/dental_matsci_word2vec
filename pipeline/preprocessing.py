from __future__ import annotations

import re
from collections import Counter
from itertools import islice
from typing import Any, Iterable, Iterator

from .config import PipelinePaths
from .example_outputs import write_preprocessed
from .io_utils import iter_jsonl, write_jsonl
from .reporting import write_stage_report

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[_-][A-Za-z0-9]+)*|\d+(?:\.\d+)?")


def _batched(iterator: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(iterator)
    while batch := list(islice(iterator, size)):
        yield batch


def _simple_tokens(text: str, lowercase: bool, remove_stopwords: bool) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    if remove_stopwords:
        from spacy.lang.en.stop_words import STOP_WORDS

        tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens


def _spacy_records(
    records: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    import spacy

    model_name = settings.get("spacy_model", "en_core_web_sm")
    try:
        nlp = spacy.load(model_name, disable=settings.get("disable", ["ner", "parser"]))
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is unavailable. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc

    nlp.max_length = int(settings.get("max_length", 3_000_000))
    batch_size = int(settings.get("batch_size", 32))
    lowercase = bool(settings.get("lowercase", True))
    remove_stopwords = bool(settings.get("remove_stopwords", True))
    stopword_source = settings.get("stopword_source", "spacy").lower()
    nltk_stopwords: set[str] = set()
    if remove_stopwords and stopword_source == "nltk":
        try:
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words("english"))
        except LookupError as exc:
            raise RuntimeError(
                "NLTK English stopwords are unavailable. Run: "
                "python -m nltk.downloader stopwords"
            ) from exc
    elif stopword_source not in {"spacy", "nltk"}:
        raise ValueError("preprocessing.stopword_source must be 'spacy' or 'nltk'.")

    lemmatize = bool(settings.get("lemmatize", True))
    remove_numbers = bool(settings.get("remove_numbers", False))

    for batch in _batched(records, batch_size):
        docs = nlp.pipe([row["text"] for row in batch], batch_size=batch_size)
        for row, doc in zip(batch, docs):
            tokens: list[str] = []
            for token in doc:
                if token.is_space or token.is_punct:
                    continue
                if remove_numbers and token.like_num:
                    continue
                if remove_stopwords:
                    raw_lower = token.text.lower()
                    if (stopword_source == "spacy" and token.is_stop) or (
                        stopword_source == "nltk" and raw_lower in nltk_stopwords
                    ):
                        continue
                value = token.lemma_ if lemmatize and token.lemma_ else token.text
                value = value.lower() if lowercase else value
                if value and value != "-PRON-":
                    tokens.append(value)
            counters["documents"] += 1
            counters["tokens"] += len(tokens)
            counters[f"documents_{row['period']}"] += 1
            counters[f"tokens_{row['period']}"] += len(tokens)
            yield {key: value for key, value in row.items() if key != "text"} | {
                "tokens": tokens
            }


def _simple_records(
    records: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    lowercase = bool(settings.get("lowercase", True))
    remove_stopwords = bool(settings.get("remove_stopwords", True))
    for row in records:
        tokens = _simple_tokens(row["text"], lowercase, remove_stopwords)
        counters["documents"] += 1
        counters["tokens"] += len(tokens)
        counters[f"documents_{row['period']}"] += 1
        counters[f"tokens_{row['period']}"] += len(tokens)
        yield {key: value for key, value in row.items() if key != "text"} | {
            "tokens": tokens
        }


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    settings = config.get("preprocessing", {})
    engine = settings.get("engine", "spacy").lower()
    counters: Counter[str] = Counter()
    records = iter_jsonl(paths.raw_documents_path)

    if engine == "spacy":
        output = _spacy_records(records, settings, counters)
    elif engine == "simple":
        output = _simple_records(records, settings, counters)
    else:
        raise ValueError("preprocessing.engine must be 'spacy' or 'simple'.")

    n_rows = write_jsonl(output, paths.preprocessed_path)
    example_result = write_preprocessed(config, paths)
    metrics = dict(counters)
    metrics["written_documents"] = n_rows
    metrics["engine"] = engine
    metrics["avg_tokens_per_document"] = (
        round(counters["tokens"] / counters["documents"], 3)
        if counters["documents"]
        else 0
    )
    write_stage_report(
        report_path=paths.reports_dir / "03_preprocessing.md",
        title="Stage 3 — Text preprocessing",
        purpose=(
            "Normalizes extracted text and writes one inspectable token list per JSONL record."
        ),
        inputs=[paths.raw_documents_path],
        outputs=[paths.preprocessed_path]
        + ([example_result[0]] if example_result else []),
        metrics=metrics,
        parameters=settings,
        notes=[
            "Use engine='spacy' for the production lemmatized corpus.",
            "The lightweight engine='simple' is intended for the bundled example and CI smoke tests.",
        ],
    )
    return metrics
