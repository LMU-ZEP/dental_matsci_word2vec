from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from gensim.models.phrases import Phrases, Phraser

from .config import PipelinePaths
from .example_outputs import write_phrased
from .io_utils import iter_jsonl, write_jsonl, write_tsv
from .reporting import write_stage_report


class JsonlTokenCorpus:
    """Re-iterable streaming token corpus used by gensim."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[list[str]]:
        for record in iter_jsonl(self.path):
            yield [str(token) for token in record.get("tokens", [])]


def _apply_phraser(
    path: Path,
    phraser: Phraser,
    counters: Counter[str],
) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(path):
        input_tokens = [str(token) for token in record.get("tokens", [])]
        output_tokens = list(phraser[input_tokens])
        counters["documents"] += 1
        counters["input_tokens"] += len(input_tokens)
        counters["output_tokens"] += len(output_tokens)
        for token in output_tokens:
            if "_" in token:
                counters[f"phrase::{token}"] += 1
                counters["phrase_tokens"] += 1
        yield record | {"tokens": output_tokens}


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    settings = config.get("phrases", {})
    corpus = JsonlTokenCorpus(paths.preprocessed_path)
    phrases = Phrases(
        corpus,
        min_count=int(settings.get("min_count", 50)),
        threshold=float(settings.get("threshold", 10.0)),
        max_vocab_size=int(settings.get("max_vocab_size", 40_000_000)),
        progress_per=int(settings.get("progress_every", 10_000)),
        delimiter=settings.get("delimiter", "_") ,
    )
    phraser = Phraser(phrases)
    paths.phraser_path.parent.mkdir(parents=True, exist_ok=True)
    phraser.save(str(paths.phraser_path))

    counters: Counter[str] = Counter()
    n_rows = write_jsonl(
        _apply_phraser(paths.preprocessed_path, phraser, counters),
        paths.phrased_path,
    )
    phrase_rows = [
        {"phrase": key.removeprefix("phrase::"), "count": count}
        for key, count in counters.most_common()
        if key.startswith("phrase::")
    ]
    write_tsv(phrase_rows, paths.phrase_counts_path, ["phrase", "count"])
    example_result = write_phrased(config, paths)

    metrics = {
        "documents": n_rows,
        "input_tokens": counters["input_tokens"],
        "output_tokens": counters["output_tokens"],
        "phrase_tokens": counters["phrase_tokens"],
        "unique_phrase_tokens": len(phrase_rows),
    }
    write_stage_report(
        report_path=paths.reports_dir / "04_shared_phraser.md",
        title="Stage 4 — Shared phrase detection",
        purpose=(
            "Fits one shared gensim Phraser on all periods and applies the same "
            "transformation to every period before model training."
        ),
        inputs=[paths.preprocessed_path],
        outputs=[paths.phraser_path, paths.phrased_path, paths.phrase_counts_path]
        + ([example_result[0]] if example_result else []),
        metrics=metrics,
        parameters=settings,
    )
    return metrics
