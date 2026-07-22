from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from gensim.models import Word2Vec

from .config import PipelinePaths
from .example_outputs import copy_vocabulary
from .io_utils import iter_jsonl, write_tsv
from .reporting import write_stage_report


class PeriodTokenCorpus:
    def __init__(self, path: str | Path, period: str):
        self.path = Path(path)
        self.period = period

    def __iter__(self) -> Iterator[list[str]]:
        for record in iter_jsonl(self.path):
            if record.get("period") == self.period:
                yield [str(token) for token in record.get("tokens", [])]


def _workers(settings: dict[str, Any]) -> int:
    if bool(settings.get("deterministic", True)):
        return 1
    if settings.get("workers") is not None:
        return max(1, int(settings["workers"]))
    return max(1, (os.cpu_count() or 1) - 1)


def _corpus_stats(corpus: PeriodTokenCorpus) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    max_tokens = 0
    for document in corpus:
        counter["documents"] += 1
        counter["tokens"] += len(document)
        max_tokens = max(max_tokens, len(document))
        if not document:
            counter["empty_documents"] += 1
    return {
        "n_documents": counter["documents"],
        "n_tokens": counter["tokens"],
        "empty_documents": counter["empty_documents"],
        "max_document_tokens": max_tokens,
        "avg_tokens_per_document": round(
            counter["tokens"] / counter["documents"], 3
        )
        if counter["documents"]
        else 0,
    }


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    settings = config.get("word2vec", {})
    seed = int(config.get("seed", settings.get("seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    workers = _workers(settings)

    vocabulary_rows: list[dict[str, Any]] = []
    period_metrics: dict[str, Any] = {}
    outputs: list[Path] = [paths.vocabulary_counts_path]

    for period in config["periods"]:
        corpus = PeriodTokenCorpus(paths.phrased_path, period)
        stats = _corpus_stats(corpus)
        if stats["n_documents"] == 0:
            raise ValueError(f"No phrased documents found for period {period!r}.")

        model = Word2Vec(
            vector_size=int(settings.get("vector_size", 200)),
            window=int(settings.get("window", 8)),
            min_count=int(settings.get("min_count", 30)),
            sample=float(settings.get("sample", 1e-4)),
            alpha=float(settings.get("alpha", 0.025)),
            min_alpha=float(settings.get("min_alpha", 0.0005)),
            negative=int(settings.get("negative", 10)),
            sg=1 if bool(settings.get("skip_gram", False)) else 0,
            workers=workers,
            seed=seed,
            sorted_vocab=1,
        )
        model.build_vocab(corpus, progress_per=int(settings.get("progress_every", 10_000)))
        if len(model.wv) == 0:
            raise ValueError(
                f"Vocabulary for period {period!r} is empty. Lower word2vec.min_count."
            )
        model.train(
            corpus,
            total_examples=model.corpus_count,
            epochs=int(settings.get("epochs", 10)),
            report_delay=float(settings.get("report_delay", 10)),
        )

        model_path = paths.model_path(period)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        outputs.append(model_path)

        for rank, token in enumerate(model.wv.index_to_key, start=1):
            vocabulary_rows.append(
                {
                    "period": period,
                    "rank": rank,
                    "token": token,
                    "count": int(model.wv.get_vecattr(token, "count")),
                }
            )

        training_config = {
            "period": period,
            "model_path": model_path.as_posix(),
            "corpus_path": paths.phrased_path.as_posix(),
            "corpus_stats": stats,
            "vocabulary_size": len(model.wv),
            "word2vec": settings | {"workers_effective": workers, "seed": seed},
        }
        config_path = paths.training_config_path(period)
        config_path.write_text(
            json.dumps(training_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs.append(config_path)
        period_metrics[period] = {
            **stats,
            "vocabulary_size": len(model.wv),
            "model_path": model_path.relative_to(paths.run_dir).as_posix(),
        }

    write_tsv(
        vocabulary_rows,
        paths.vocabulary_counts_path,
        ["period", "rank", "token", "count"],
    )
    example_path = copy_vocabulary(config, paths)
    if example_path:
        outputs.append(example_path)
    metrics = {"periods": period_metrics, "workers_effective": workers, "seed": seed}
    write_stage_report(
        report_path=paths.reports_dir / "05_word2vec_training.md",
        title="Stage 5 — Period-specific Word2Vec training",
        purpose=(
            "Trains one model per period from corpora transformed by the same shared phraser."
        ),
        inputs=[paths.phrased_path, paths.phraser_path],
        outputs=outputs,
        metrics=metrics,
        parameters=settings,
        notes=[
            "deterministic=true forces workers=1; this is required for reproducible gensim training.",
        ],
    )
    return metrics
