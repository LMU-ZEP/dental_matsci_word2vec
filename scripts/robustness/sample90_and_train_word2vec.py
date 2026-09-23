#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, os, random
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
        with self.path.open('r', encoding='utf-8') as f:
            for doc in ijson.items(f, 'item'):
                if isinstance(doc, list):
                    yield [str(t) for t in doc]
                elif isinstance(doc, str):
                    yield doc.split()
                else:
                    yield [str(doc)]


def count_units(path: Path) -> int:
    require_ijson()
    n = 0
    with path.open('r', encoding='utf-8') as f:
        for _ in ijson.items(f, 'item'):
            n += 1
    return n


def sample_indices(n_units: int, fraction: float, seed: int) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError(f'fraction must be in (0, 1], got {fraction}')
    k = int(round(n_units * fraction))
    rng = random.Random(seed)
    idx = rng.sample(range(n_units), k=k)
    idx.sort()
    return idx


def save_indices(indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for i in indices:
            f.write(f'{i}\n')


def write_sampled_corpus(input_path: Path, output_path: Path, selected: list[int], progress_every: int) -> dict:
    require_ijson()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected)
    n_seen = n_written = n_tokens = empty = max_tokens = 0
    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        f_out.write('[\n')
        first = True
        for i, doc in enumerate(ijson.items(f_in, 'item')):
            n_seen += 1
            if i not in selected_set:
                continue
            if isinstance(doc, list):
                tokens = [str(t) for t in doc]
            elif isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = [str(doc)]
            if not first:
                f_out.write(',\n')
            json.dump(tokens, f_out, ensure_ascii=False)
            first = False
            n_written += 1
            n = len(tokens)
            n_tokens += n
            max_tokens = max(max_tokens, n)
            if n == 0:
                empty += 1
            if n_written == 1 or n_written % progress_every == 0:
                print(f'  wrote {n_written} sampled corpus units from {input_path}', flush=True)
        f_out.write('\n]')
    return {
        'input_path': input_path.as_posix(),
        'output_path': output_path.as_posix(),
        'n_seen': n_seen,
        'n_written': n_written,
        'n_tokens': n_tokens,
        'empty_docs': empty,
        'max_doc_tokens': max_tokens,
        'avg_tokens_per_unit': round(n_tokens / n_written, 3) if n_written else 0,
    }


def save_vocab(model: Word2Vec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['token', 'count'])
        for tok in model.wv.index_to_key:
            w.writerow([tok, model.wv.get_vecattr(tok, 'count')])


def get_workers(deterministic: bool, workers: int | None) -> int:
    if deterministic:
        return 1
    if workers is not None:
        return max(1, workers)
    return max(1, (os.cpu_count() or 1) - 1)


def train_w2v(corpus_path: Path, model_path: Path, vocab_path: Path, args) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    workers = get_workers(args.deterministic, args.workers)
    corpus = JsonTokenCorpus(corpus_path)
    print(f'Training Word2Vec on {corpus_path}', flush=True)
    print(f'  workers: {workers}', flush=True)
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
    print(f'  vocabulary size: {len(model.wv)}', flush=True)
    print(f'  corpus_count: {model.corpus_count}', flush=True)
    print(f'  corpus_total_words: {model.corpus_total_words}', flush=True)
    model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs, report_delay=10)
    model.save(str(model_path))
    save_vocab(model, vocab_path)
    return {
        'corpus_path': corpus_path.as_posix(),
        'model_path': model_path.as_posix(),
        'vocab_path': vocab_path.as_posix(),
        'vocabulary_size': len(model.wv),
        'corpus_count': model.corpus_count,
        'corpus_total_words': model.corpus_total_words,
        'workers': workers,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Sample corpus units and train Word2Vec consistency models.')
    p.add_argument('--input-corpora', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fraction', type=float, default=0.90)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seed-offset-per-corpus', type=int, default=1000)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--progress-every', type=int, default=10_000)
    p.add_argument('--vector-size', type=int, default=200)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--sample', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.025)
    p.add_argument('--min-alpha', type=float, default=0.0005)
    p.add_argument('--negative', type=int, default=15)
    architecture = p.add_mutually_exclusive_group()
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
    p.set_defaults(skip_gram=True)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    paths = [Path(p) for p in args.input_corpora]
    if len(paths) != len(args.labels):
        raise ValueError('--input-corpora and --labels must have the same length')
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    out = Path(args.output_dir)
    sampled_dir = out / 'sampled_corpora'
    indices_dir = out / 'sampled_indices'
    models_dir = out / 'models'
    configs_dir = out / 'configs'
    for d in [sampled_dir, indices_dir, models_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    summaries = []
    for j, (path, label) in enumerate(zip(paths, args.labels)):
        seed = args.seed + j * args.seed_offset_per_corpus
        print('\n' + '='*80, flush=True)
        print(f'Processing {label}', flush=True)
        print('='*80, flush=True)
        n_units = count_units(path)
        print(f'total corpus units: {n_units}', flush=True)
        selected = sample_indices(n_units, args.fraction, seed)
        indices_path = indices_dir / f'{label}_fraction{args.fraction:.2f}_seed{seed}.indices.txt'
        save_indices(selected, indices_path)
        print(f'selected corpus units: {len(selected)}', flush=True)
        sampled_path = sampled_dir / f'{label}_fraction{args.fraction:.2f}_seed{seed}.shared_phrased.json'
        sample_stats = write_sampled_corpus(path, sampled_path, selected, args.progress_every)
        print('sample stats:', json.dumps(sample_stats, ensure_ascii=False, indent=2), flush=True)
        model_path = models_dir / f'{label}_fraction{args.fraction:.2f}_seed{seed}.model'
        vocab_path = models_dir / f'{label}_fraction{args.fraction:.2f}_seed{seed}.vocab.csv'
        train_stats = train_w2v(sampled_path, model_path, vocab_path, args)
        summary = {
            'label': label,
            'input_corpus': path.as_posix(),
            'sampled_corpus': sampled_path.as_posix(),
            'selected_indices_path': indices_path.as_posix(),
            'fraction': args.fraction,
            'n_units_total': n_units,
            'n_units_selected': len(selected),
            'sampling_seed': seed,
            'sample_stats': sample_stats,
            'train_stats': train_stats,
            'word2vec_params': {
                'vector_size': args.vector_size,
                'window': args.window,
                'min_count': args.min_count,
                'epochs': args.epochs,
                'sample': args.sample,
                'negative': args.negative,
                'alpha': args.alpha,
                'min_alpha': args.min_alpha,
                'skip_gram': bool(args.skip_gram),
                'sg': 1 if args.skip_gram else 0,
                'seed': args.seed,
                'deterministic': bool(args.deterministic),
                'workers': get_workers(args.deterministic, args.workers),
            },
        }
        config_path = configs_dir / f'{label}_fraction{args.fraction:.2f}_seed{seed}.config.json'
        config_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        summaries.append(summary)
    summary_path = out / f'sample_train_summary_fraction{args.fraction:.2f}_seed{args.seed}.json'
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved run summary: {summary_path}', flush=True)
    print('Done.', flush=True)

if __name__ == '__main__':
    main()
