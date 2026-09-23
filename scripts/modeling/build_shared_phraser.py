#!/usr/bin/env python3
"""
Build one shared gensim bigram phraser on multiple preprocessed corpora
and apply it to each corpus separately.

Input format:
    JSON list of token lists:
    [
      ["flexural", "strength", "of", "zirconia"],
      ["resin", "composite", "surface", "roughness"]
    ]

Output format:
    JSON list of token lists after shared phrase detection:
    [
      ["flexural_strength", "zirconia"],
      ["resin_composite", "surface_roughness"]
    ]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import ijson
except ImportError:
    ijson = None
from gensim.models.phrases import Phrases, Phraser



def require_ijson() -> None:
    if ijson is None:
        raise RuntimeError(
            "Missing required dependency 'ijson'. Install repository requirements before running data processing/training."
        )


class JsonTokenCorpus:
    """Re-iterable streaming corpus over one JSON list of token lists."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterable[list[str]]:
        require_ijson()
        with self.path.open("r", encoding="utf-8") as f:
            for doc in ijson.items(f, "item"):
                if isinstance(doc, list):
                    yield [str(tok) for tok in doc]
                elif isinstance(doc, str):
                    # Defensive fallback. Intended input is token lists.
                    yield doc.split()
                else:
                    yield [str(doc)]


class MultiJsonTokenCorpus:
    """
    Re-iterable streaming corpus over several JSON corpora.
    The corpora are read sequentially and never loaded fully into memory.
    """

    def __init__(self, paths: list[str | Path]):
        self.paths = [Path(p) for p in paths]

    def __iter__(self) -> Iterable[list[str]]:
        for path in self.paths:
            for doc in JsonTokenCorpus(path):
                yield doc


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_phraser_to_corpus(
    *,
    input_path: str | Path,
    output_path: str | Path,
    phraser: Phraser,
    progress_every: int = 10_000,
) -> dict:
    """
    Apply phraser to one corpus and write a new JSON list of token lists.

    Returns a summary dictionary. Phrase-token counts are returned as a Counter
    under the temporary key "_phrase_counter".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    input_tokens = 0
    output_tokens = 0
    phrase_tokens_total = 0
    phrase_counter: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        f_out.write("[\n")
        first = True

        for doc in ijson.items(f_in, "item"):
            n_docs += 1

            if isinstance(doc, list):
                tokens = [str(tok) for tok in doc]
            elif isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = [str(doc)]

            phrased = list(phraser[tokens])

            input_tokens += len(tokens)
            output_tokens += len(phrased)

            for tok in phrased:
                if "_" in tok:
                    phrase_counter[tok] += 1
                    phrase_tokens_total += 1

            if not first:
                f_out.write(",\n")
            json.dump(phrased, f_out, ensure_ascii=False)
            first = False

            if n_docs == 1 or n_docs % progress_every == 0:
                print(f"  applied to {n_docs} documents from {input_path}")

        f_out.write("\n]")

    return {
        "input_path": input_path.as_posix(),
        "output_path": output_path.as_posix(),
        "n_docs": n_docs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "phrase_tokens_total": phrase_tokens_total,
        "unique_phrase_tokens": len(phrase_counter),
        "_phrase_counter": phrase_counter,
    }


def save_top_phrases(
    *,
    phrase_counters: dict[str, Counter[str]],
    output_path: str | Path,
    topn: int,
) -> None:
    rows: list[dict] = []

    combined: Counter[str] = Counter()
    for counter in phrase_counters.values():
        combined.update(counter)

    for rank, (phrase, count) in enumerate(combined.most_common(topn), start=1):
        row = {
            "rank": rank,
            "phrase": phrase,
            "combined_count": count,
        }
        for corpus_label, counter in phrase_counters.items():
            row[f"{corpus_label}_count"] = counter.get(phrase, 0)
        rows.append(row)

    write_csv(rows, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared gensim Phraser on multiple preprocessed corpora "
            "and apply it to each corpus separately."
        )
    )

    parser.add_argument(
        "--input-corpora",
        nargs="+",
        required=True,
        help="Input preprocessed JSON corpora, each a JSON list of token lists.",
    )
    parser.add_argument(
        "--output-corpora",
        nargs="+",
        required=True,
        help=(
            "Output phrased JSON corpora. Must have the same number and order "
            "as --input-corpora."
        ),
    )
    parser.add_argument(
        "--phraser-path",
        required=True,
        help="Where to save the shared gensim Phraser.",
    )
    parser.add_argument(
        "--phrase-min-count",
        type=int,
        default=30,
        help="Minimum bigram count for phrase detection.",
    )
    parser.add_argument(
        "--phrase-threshold",
        type=float,
        default=10.0,
        help="Gensim Phrases threshold. Higher values create fewer phrases.",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=40_000_000,
        help="Maximum vocabulary size for gensim Phrases.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N documents.",
    )
    parser.add_argument(
        "--top-phrases-csv",
        default=None,
        help="Optional path to write top phrase token counts.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write run summary JSON.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=5000,
        help="Number of top phrase tokens to write to CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(p) for p in args.input_corpora]
    output_paths = [Path(p) for p in args.output_corpora]
    phraser_path = Path(args.phraser_path)

    if len(input_paths) != len(output_paths):
        raise ValueError(
            "--input-corpora and --output-corpora must have the same length. "
            f"Got {len(input_paths)} inputs and {len(output_paths)} outputs."
        )

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    phraser_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training shared bigram phraser")
    print("Input corpora:")
    for p in input_paths:
        print(f"  - {p}")
    print(f"phrase_min_count: {args.phrase_min_count}")
    print(f"phrase_threshold: {args.phrase_threshold}")

    shared_corpus = MultiJsonTokenCorpus(input_paths)

    phrases = Phrases(
        shared_corpus,
        min_count=args.phrase_min_count,
        threshold=args.phrase_threshold,
        max_vocab_size=args.max_vocab_size,
        progress_per=args.progress_every,
    )

    phraser = Phraser(phrases)
    phraser.save(str(phraser_path))
    print(f"Saved shared phraser to: {phraser_path}")

    print("\nApplying shared phraser to each corpus")
    summaries: list[dict] = []
    phrase_counters: dict[str, Counter[str]] = {}

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"\nInput:  {input_path}")
        print(f"Output: {output_path}")

        summary = apply_phraser_to_corpus(
            input_path=input_path,
            output_path=output_path,
            phraser=phraser,
            progress_every=args.progress_every,
        )

        counter = summary.pop("_phrase_counter")
        label = input_path.stem
        phrase_counters[label] = counter
        summaries.append(summary)

        print(
            "Done:",
            f"{summary['n_docs']} docs,",
            f"{summary['input_tokens']} input tokens,",
            f"{summary['output_tokens']} output tokens,",
            f"{summary['unique_phrase_tokens']} unique phrase tokens",
        )

    top_phrases_csv = (
        Path(args.top_phrases_csv)
        if args.top_phrases_csv
        else phraser_path.with_suffix(".top_phrases.csv")
    )
    save_top_phrases(
        phrase_counters=phrase_counters,
        output_path=top_phrases_csv,
        topn=args.topn,
    )
    print(f"\nSaved top phrase counts to: {top_phrases_csv}")

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else phraser_path.with_suffix(".summary.json")
    )
    summary = {
        "input_corpora": [p.as_posix() for p in input_paths],
        "output_corpora": [p.as_posix() for p in output_paths],
        "phraser_path": phraser_path.as_posix(),
        "phrase_min_count": args.phrase_min_count,
        "phrase_threshold": args.phrase_threshold,
        "max_vocab_size": args.max_vocab_size,
        "corpus_summaries": summaries,
        "top_phrases_csv": top_phrases_csv.as_posix(),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary to: {summary_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
