#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

try:
    import ijson
except ImportError:
    ijson = None
import numpy as np
from gensim.models import Word2Vec



def require_ijson() -> None:
    if ijson is None:
        raise RuntimeError(
            "Missing required dependency 'ijson'. Install repository requirements before running data processing/training."
        )


class JsonTokenCorpus:
    """Re-iterable streaming corpus over a JSON list of token lists."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterable[list[str]]:
        require_ijson()
        with self.path.open("r", encoding="utf-8") as f:
            for doc in ijson.items(f, "item"):
                if isinstance(doc, list):
                    yield [str(tok) for tok in doc]
                elif isinstance(doc, str):
                    yield doc.split()
                else:
                    yield [str(doc)]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def effective_workers(deterministic: bool, workers: int | None) -> int:
    if deterministic:
        return 1
    if workers is not None:
        return max(1, workers)
    return max(1, (os.cpu_count() or 1) - 1)


def corpus_stats(path: str | Path) -> dict:
    n_docs = 0
    n_tokens = 0
    empty_docs = 0
    max_doc_tokens = 0

    for doc in JsonTokenCorpus(path):
        n_docs += 1
        n = len(doc)
        n_tokens += n
        max_doc_tokens = max(max_doc_tokens, n)
        if n == 0:
            empty_docs += 1

    return {
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "empty_docs": empty_docs,
        "max_doc_tokens": max_doc_tokens,
        "avg_tokens_per_doc": round(n_tokens / n_docs, 3) if n_docs else 0,
    }


def save_vocab(model: Word2Vec, vocab_path: str | Path) -> None:
    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for token in model.wv.index_to_key:
            writer.writerow([token, model.wv.get_vecattr(token, "count")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Word2Vec directly from an already tokenized JSON corpus."
    )

    parser.add_argument("--corpus", required=True, help="JSON list of token lists.")
    parser.add_argument("--model-path", required=True, help="Output gensim Word2Vec model path.")
    parser.add_argument("--vocab-path", default=None, help="Output vocabulary CSV path.")
    parser.add_argument("--config-path", default=None, help="Output training config JSON path.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Force workers=1.")
    parser.add_argument("--workers", type=int, default=None)

    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--min-alpha", type=float, default=0.0005)
    parser.add_argument("--negative", type=int, default=15)
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument(
        "--skip-gram",
        dest="skip_gram",
        action="store_true",
        help="Use Skip-Gram architecture (default).",
    )
    architecture.add_argument(
        "--cbow",
        dest="skip_gram",
        action="store_false",
        help="Use CBOW architecture instead of Skip-Gram.",
    )
    parser.set_defaults(skip_gram=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    corpus_path = Path(args.corpus)
    model_path = Path(args.model_path)
    vocab_path = Path(args.vocab_path) if args.vocab_path else model_path.with_suffix(".vocab.csv")
    config_path = Path(args.config_path) if args.config_path else model_path.with_suffix(".config.json")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing corpus statistics")
    stats = corpus_stats(corpus_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    corpus = JsonTokenCorpus(corpus_path)
    workers = effective_workers(args.deterministic, args.workers)

    print("Training Word2Vec")
    print(f"corpus: {corpus_path}")
    print(f"model_path: {model_path}")
    print(f"workers: {workers}")
    print(f"skip_gram: {args.skip_gram}")

    model = Word2Vec(
        min_count=args.min_count,
        window=args.window,
        vector_size=args.vector_size,
        sample=args.sample,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        negative=args.negative,
        sg=1 if args.skip_gram else 0,
        workers=workers,
        seed=args.seed,
    )

    model.build_vocab(corpus, progress_per=10_000)

    print(f"vocabulary size: {len(model.wv)}")
    print(f"corpus_count: {model.corpus_count}")
    print(f"corpus_total_words: {model.corpus_total_words}")

    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        report_delay=10,
    )

    model.save(str(model_path))
    save_vocab(model, vocab_path)

    config = {
        "corpus": corpus_path.as_posix(),
        "model_path": model_path.as_posix(),
        "vocab_path": vocab_path.as_posix(),
        "config_path": config_path.as_posix(),
        "corpus_stats": stats,
        "word2vec": {
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "sample": args.sample,
            "alpha": args.alpha,
            "min_alpha": args.min_alpha,
            "negative": args.negative,
            "sg": 1 if args.skip_gram else 0,
            "skip_gram": bool(args.skip_gram),
            "seed": args.seed,
            "workers": workers,
            "deterministic": bool(args.deterministic),
        },
        "phrase_detection": {
            "trained_in_this_script": False,
            "note": "This script expects phrase detection to be already applied, e.g. by build_shared_phraser.py.",
        },
    }

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved vocab: {vocab_path}")
    print(f"Saved config: {config_path}")
    print("Done.")


if __name__ == "__main__":
    main()
