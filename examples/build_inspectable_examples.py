#!/usr/bin/env python3
"""Build a small, fully redistributable set of inspectable pipeline intermediates.

The input fixtures are synthetic and contain no article text. The script deliberately
uses the repository's actual XML/PDF extraction, production spaCy/NLTK preprocessing,
shared phrase-detection, Word2Vec, Procrustes, and nearest-neighbor/Jaccard code.

Demo-model parameters are scaled down for the tiny fixture; they are NOT the manuscript
training parameters. Canonical manuscript parameters remain in run_metadata/ and README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FIXTURES = EXAMPLES / "fixtures"
DEFAULT_OUT = EXAMPLES / "inspectable_pipeline"

DEMO_WORD2VEC = {
    "vector_size": 30,
    "window": 5,
    "min_count": 1,
    "epochs": 20,
    "sample": 0.001,
    "negative": 5,
    "alpha": 0.025,
    "min_alpha": 0.0005,
    "skip_gram": True,
    "seed": 42,
    "workers": 1,
}
DEMO_PHRASES = {"min_count": 2, "threshold": 0.1}


def run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode:
        raise RuntimeError(
            "COMMAND FAILED\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + p.stdout
            + "\nSTDERR:\n"
            + p.stderr
        )
    return p


def ensure_production_resources() -> None:
    missing: list[str] = []
    try:
        import ijson  # noqa: F401
    except ImportError:
        missing.append("ijson")
    try:
        import spacy
        spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except (ImportError, OSError):
        missing.append("spaCy model en_core_web_sm")
    try:
        from nltk.corpus import stopwords
        stopwords.words("english")
    except (ImportError, LookupError):
        missing.append("NLTK stopwords")
    if missing:
        raise RuntimeError(
            "Cannot build canonical inspectable examples because production resources are missing: "
            + ", ".join(missing)
            + ". Install repository requirements/resources first."
        )


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)

    # Register the module before executing it. Decorators such as @dataclass
    # inspect sys.modules[cls.__module__] while the module body is executing.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_paths(obj, prefixes: list[tuple[str, str]]):
    if isinstance(obj, dict):
        return {k: sanitize_paths(v, prefixes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_paths(v, prefixes) for v in obj]
    if isinstance(obj, str):
        value = obj
        for old, new in prefixes:
            if value.startswith(old):
                value = new + value[len(old) :]
        return value
    return obj


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_stats(corpus: list[list[str]]) -> dict:
    counts = [len(x) for x in corpus]
    return {
        "n_units": len(corpus),
        "n_tokens": sum(counts),
        "empty_units": sum(n == 0 for n in counts),
        "max_tokens_per_unit": max(counts, default=0),
        "mean_tokens_per_unit": round(sum(counts) / len(counts), 3) if counts else 0,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows: list[list[object]], headers: list[str]) -> str:
    if not rows:
        return "(no rows)\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    ensure_production_resources()

    out = Path(args.out_dir).resolve()
    if out == ROOT or ROOT in out.parents and out.name in {"fixtures", "manuscript_output_snapshots"}:
        raise ValueError("Refusing unsafe output directory")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"

    d01 = out / "01_sources_and_extraction"
    d02 = out / "02_preprocessing"
    d03 = out / "03_shared_phrases"
    d04 = out / "04_word2vec"
    d05 = out / "05_procrustes"
    d06 = out / "06_semantic_evidence"
    reports = out / "reports"
    for d in [d01, d02, d03, d04, d05, d06, reports]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Synthetic source manifest + actual XML/PDF extraction
    # ------------------------------------------------------------------
    source_rows: list[dict] = []
    for period, subdir in [("pre2018", "xml_pre"), ("post2018", "xml_post")]:
        for p in sorted((FIXTURES / subdir).glob("*.xml")):
            source_rows.append(
                {
                    "period": period,
                    "source_type": "xml",
                    "path": p.relative_to(ROOT).as_posix(),
                    "synthetic": True,
                    "included_in_demo_training": True,
                }
            )
    for period, filename in [
        ("pre2018", "synthetic_pre2018.pdf"),
        ("post2018", "synthetic_post2018.pdf"),
    ]:
        p = FIXTURES / "pdf" / filename
        source_rows.append(
            {
                "period": period,
                "source_type": "pdf",
                "path": p.relative_to(ROOT).as_posix(),
                "synthetic": True,
                "included_in_demo_training": True,
            }
        )

    with (d01 / "source_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in source_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    xml_raw_paths: dict[str, Path] = {}
    for label, subdir, min_year in [
        ("pre", "xml_pre", "1992"),
        ("post", "xml_post", "2018"),
    ]:
        target = d01 / f"{label}_xml_raw.json"
        run(
            [
                sys.executable,
                str(ROOT / "scripts/corpus/get_xml_corpus.py"),
                "--xml-inputs",
                str(FIXTURES / subdir),
                "--output-json",
                str(target),
                "--min-year",
                min_year,
            ],
            env=env,
        )
        xml_raw_paths[label] = target

    # PDF extraction through actual production helper.
    sys.path.insert(0, str(ROOT / "scripts/corpus"))
    pdf_mod = import_from_path("repo_pdf_extraction", ROOT / "scripts/corpus/pdf_extraction.py")
    pdf_results: dict[str, dict] = {}
    for label, filename in [
        ("pre", "synthetic_pre2018.pdf"),
        ("post", "synthetic_post2018.pdf"),
    ]:
        p = FIXTURES / "pdf" / filename
        res = pdf_mod.extract_pdf_text(p, backend="pymupdf_columns", fallback_backend="pypdf")
        if res.status not in {"ok", "ok_with_warnings"} or not res.cleaned_text.strip():
            raise RuntimeError(f"PDF extraction failed for {p}: {res.status}: {res.error}")
        record = {
            "fixture": p.relative_to(ROOT).as_posix(),
            "requested_backend": res.requested_backend,
            "backend_used": res.backend_used,
            "status": res.status,
            "fallback_used": res.fallback_used,
            "n_pages": res.n_pages,
            "raw_chars": res.raw_chars,
            "clean_chars": res.clean_chars,
            "citation_markers_before_cleanup": res.citation_markers_before_cleanup,
            "error": res.error,
            "raw_text": res.raw_text,
            "cleaned_text": res.cleaned_text,
        }
        pdf_results[label] = record
        write_json(d01 / f"{label}_pdf_extraction.json", record)

    # Combine XML chunks and one synthetic PDF extraction per period. This is a tiny
    # demonstration corpus, not a reproduction of source-selection counts.
    raw_paths: dict[str, Path] = {}
    for label in ["pre", "post"]:
        xml_items = load_json(xml_raw_paths[label])
        combined = list(xml_items) + [pdf_results[label]["cleaned_text"]]
        target = d01 / f"{label}_raw_combined.json"
        write_json(target, combined)
        raw_paths[label] = target

    # ------------------------------------------------------------------
    # 2. Actual production preprocessing from raw text to token lists
    # ------------------------------------------------------------------
    prep_paths: dict[str, Path] = {}
    for label in ["pre", "post"]:
        target = d02 / f"{label}_preprocessed.json"
        run(
            [
                sys.executable,
                str(ROOT / "scripts/corpus/word2vec_pipeline_step2_xml_pdf.py"),
                "--corpus-name",
                "inspectable_demo",
                "--period",
                label,
                "--reuse-raw-corpus",
                "--raw-corpus-path",
                str(raw_paths[label]),
                "--preprocessed-corpus-path",
                str(target),
                "--output-dir",
                str(d02 / f"run_{label}"),
            ],
            env=env,
        )
        prep_paths[label] = target

    pre_prep = load_json(prep_paths["pre"])
    post_prep = load_json(prep_paths["post"])
    if not pre_prep or not post_prep:
        raise AssertionError("Production preprocessing produced an empty demo corpus")
    preprocessing_summary = {
        "pre": token_stats(pre_prep),
        "post": token_stats(post_prep),
        "note": "Generated by the actual production spaCy/NLTK preprocessing code from synthetic source text.",
    }
    write_json(d02 / "preprocessing_summary.json", preprocessing_summary)

    # ------------------------------------------------------------------
    # 3. Actual shared phrase stage, with DEMO-SCALED parameters
    # ------------------------------------------------------------------
    phrase_summary = d03 / "shared_phraser_summary.json"
    top_phrases = d03 / "top_phrases.csv"
    phraser = d03 / "shared_demo.phraser"
    phrased = {"pre": d03 / "pre_shared_phrased.json", "post": d03 / "post_shared_phrased.json"}
    run(
        [
            sys.executable,
            str(ROOT / "scripts/modeling/build_shared_phraser.py"),
            "--input-corpora",
            str(prep_paths["pre"]),
            str(prep_paths["post"]),
            "--output-corpora",
            str(phrased["pre"]),
            str(phrased["post"]),
            "--phraser-path",
            str(phraser),
            "--phrase-min-count",
            str(DEMO_PHRASES["min_count"]),
            "--phrase-threshold",
            str(DEMO_PHRASES["threshold"]),
            "--top-phrases-csv",
            str(top_phrases),
            "--summary-json",
            str(phrase_summary),
            "--topn",
            "100",
        ],
        env=env,
    )

    # ------------------------------------------------------------------
    # 4. Actual Word2Vec training on the demo corpora
    # ------------------------------------------------------------------
    model_paths: dict[str, Path] = {}
    vocab_paths: dict[str, Path] = {}
    config_paths: dict[str, Path] = {}
    for label in ["pre", "post"]:
        model = d04 / f"{label}_demo.model"
        vocab = d04 / f"{label}_demo.vocab.csv"
        config = d04 / f"{label}_demo.config.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/modeling/train_word2vec_from_tokens.py"),
            "--corpus",
            str(phrased[label]),
            "--model-path",
            str(model),
            "--vocab-path",
            str(vocab),
            "--config-path",
            str(config),
            "--seed",
            "42",
            "--deterministic",
            "--vector-size",
            str(DEMO_WORD2VEC["vector_size"]),
            "--window",
            str(DEMO_WORD2VEC["window"]),
            "--min-count",
            str(DEMO_WORD2VEC["min_count"]),
            "--epochs",
            str(DEMO_WORD2VEC["epochs"]),
            "--sample",
            str(DEMO_WORD2VEC["sample"]),
            "--negative",
            str(DEMO_WORD2VEC["negative"]),
            "--alpha",
            str(DEMO_WORD2VEC["alpha"]),
            "--min-alpha",
            str(DEMO_WORD2VEC["min_alpha"]),
            "--skip-gram",
        ]
        run(cmd, env=env)
        model_paths[label] = model
        vocab_paths[label] = vocab
        config_paths[label] = config

    # Sanitize environment-specific paths in committed example JSON configs.
    prefixes = [(str(ROOT), "<REPOSITORY_ROOT>"), (str(out), "<EXAMPLE_OUTPUT>")]
    for p in [phrase_summary, *config_paths.values()]:
        obj = load_json(p)
        write_json(p, sanitize_paths(obj, prefixes))

    # Export a human-inspectable sample of trained vector coordinates.
    from gensim.models import Word2Vec

    models = {label: Word2Vec.load(str(path)) for label, path in model_paths.items()}
    common = set(models["pre"].wv.index_to_key) & set(models["post"].wv.index_to_key)
    clean_common = [
        w
        for w in common
        if len(w) >= 3 and re.search(r"[A-Za-z]", w) and not re.fullmatch(r"\d+", w)
    ]
    clean_common.sort(
        key=lambda w: (
            -min(
                int(models["pre"].wv.get_vecattr(w, "count")),
                int(models["post"].wv.get_vecattr(w, "count")),
            ),
            w,
        )
    )
    vector_terms = clean_common[:12]
    with (d04 / "trained_vectors_sample.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["period", "term", "count"] + [f"dim_{i}" for i in range(8)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label in ["pre", "post"]:
            m = models[label]
            for term in vector_terms:
                vec = m.wv[term]
                row = {
                    "period": label,
                    "term": term,
                    "count": int(m.wv.get_vecattr(term, "count")),
                }
                row.update({f"dim_{i}": float(vec[i]) for i in range(8)})
                writer.writerow(row)

    # ------------------------------------------------------------------
    # 5. Actual Procrustes displacement with a small target/background set
    # ------------------------------------------------------------------
    preferred = [
        "fracture_toughness",
        "crack_bridging",
        "shade_matching",
        "color_matching",
        "zirconia",
        "strength",
        "optical",
        "resin_composite",
        "fracture",
        "shade",
        "color",
    ]
    selected: list[str] = []
    for term in preferred + clean_common:
        if term in common and term not in selected:
            selected.append(term)
        if len(selected) == 3:
            break
    background = [w for w in clean_common if w not in selected]
    if len(selected) < 3 or len(background) < 3:
        raise AssertionError(
            f"Demo vocabulary too small: common={len(common)}, selected={selected}, background={background[:20]}"
        )

    groups = ["zirconia", "short_fiber_composite", "structural_color_composite"]
    target_tsv = d05 / "demo_target_terms.tsv"
    with target_tsv.open("w", encoding="utf-8") as f:
        f.write("term\tmaterial_system\n")
        for term, group in zip(selected, groups):
            f.write(f"{term}\t{group}\n")

    run(
        [
            sys.executable,
            str(ROOT / "scripts/analysis/procrustes_displacement.py"),
            "--pre-model",
            str(model_paths["pre"]),
            "--post-model",
            str(model_paths["post"]),
            "--target-terms",
            str(target_tsv),
            "--out-dir",
            str(d05),
            "--min-count",
            "1",
            "--top-n",
            "15",
        ],
        env=env,
    )
    proc_meta = load_json(d05 / "procrustes_metadata.json")
    write_json(d05 / "procrustes_metadata.json", sanitize_paths(proc_meta, prefixes))

    # ------------------------------------------------------------------
    # 6. Actual nearest-neighbor/Jaccard code
    # ------------------------------------------------------------------
    sem = import_from_path(
        "repo_semantic_shift_evidence", ROOT / "scripts/analysis/semantic_shift_evidence.py"
    )
    terms_df = sem.read_terms_table(target_tsv)
    pre_wv = sem.load_wv(model_paths["pre"])
    post_wv = sem.load_wv(model_paths["post"])
    sem.compute_jaccard_and_neighbors(
        pre_wv,
        post_wv,
        terms_df,
        d06,
        k_values=(5, 10),
        nn_topn=10,
        mode="common_vocab",
    )

    # ------------------------------------------------------------------
    # Human-readable stage reports
    # ------------------------------------------------------------------
    pre_raw = load_json(raw_paths["pre"])
    post_raw = load_json(raw_paths["post"])
    report1 = f"""# 01. Sources and extraction

All inputs in this demonstration are synthetic and redistributable. They are **not manuscript articles**.

- synthetic XML files: 4 pre-2018 + 4 post-2018
- synthetic PDF files: 1 pre-2018 + 1 post-2018
- combined raw text units: {len(pre_raw)} pre / {len(post_raw)} post
- PDF backend used: {pdf_results['pre']['backend_used']} (pre), {pdf_results['post']['backend_used']} (post)

The PDF extraction JSON files expose both `raw_text` and the normalized `cleaned_text` so extraction/cleanup can be inspected directly.
"""
    (reports / "01_sources_and_extraction.md").write_text(report1, encoding="utf-8")

    pre_stats = preprocessing_summary["pre"]
    post_stats = preprocessing_summary["post"]
    preview_pre = " ".join(pre_prep[0][:30]) if pre_prep else ""
    preview_post = " ".join(post_prep[0][:30]) if post_prep else ""
    report2 = f"""# 02. Production preprocessing

This stage was generated by `scripts/corpus/word2vec_pipeline_step2_xml_pdf.py` using the same spaCy/NLTK preprocessing implementation as the manuscript pipeline.

| period | units | tokens | mean tokens/unit |
| --- | ---: | ---: | ---: |
| pre | {pre_stats['n_units']} | {pre_stats['n_tokens']} | {pre_stats['mean_tokens_per_unit']} |
| post | {post_stats['n_units']} | {post_stats['n_tokens']} | {post_stats['mean_tokens_per_unit']} |

First pre unit (first 30 tokens): `{preview_pre}`

First post unit (first 30 tokens): `{preview_post}`
"""
    (reports / "02_preprocessing.md").write_text(report2, encoding="utf-8")

    ps = load_json(phrase_summary)
    phrase_rows: list[list[object]] = []
    if top_phrases.exists() and top_phrases.stat().st_size:
        with top_phrases.open(encoding="utf-8", newline="") as f:
            for row in list(csv.DictReader(f))[:10]:
                phrase_rows.append([row.get("rank", ""), row.get("phrase", ""), row.get("combined_count", "")])
    report3 = f"""# 03. Shared phrase detection

The actual shared-phraser implementation is used, but the tiny demonstration requires relaxed parameters:

- demo `min_count = {DEMO_PHRASES['min_count']}`
- demo `threshold = {DEMO_PHRASES['threshold']}`

These are **not** the manuscript values (`min_count=30`, `threshold=10.0`). The scientific requirement is the same: one phraser is trained on the combined pre/post corpora and applied unchanged to both periods.

Top phrase tokens in this demo:

{md_table(phrase_rows, ['rank','phrase','combined count'])}
"""
    (reports / "03_shared_phrases.md").write_text(report3, encoding="utf-8")

    pre_conf = load_json(config_paths["pre"])
    post_conf = load_json(config_paths["post"])
    report4 = f"""# 04. Word2Vec training

The actual deterministic Word2Vec training script is used. Parameters are scaled down only for this tiny fixture.

```json
{json.dumps(DEMO_WORD2VEC, indent=2)}
```

Vocabulary sizes:

- pre demo model: {sum(1 for _ in csv.DictReader(vocab_paths['pre'].open(encoding='utf-8')))}
- post demo model: {sum(1 for _ in csv.DictReader(vocab_paths['post'].open(encoding='utf-8')))}

`trained_vectors_sample.csv` exposes the first eight coordinates for 12 shared terms. Full tiny Gensim model files are also retained for exact downstream inspection.
"""
    (reports / "04_word2vec_training.md").write_text(report4, encoding="utf-8")

    import pandas as pd
    disp = pd.read_csv(d05 / "cosine_displacement.csv")
    rows5 = [[r.term, round(float(r.cosine_displacement), 6), int(r.pre_count), int(r.post_count)] for r in disp.itertuples()]
    report5 = f"""# 05. Procrustes alignment and displacement

The actual `procrustes_displacement.py` is run on the two tiny demonstration models.
The alignment vocabulary is deliberately small (`top_n=15`, `min_count=1`) because the fixture contains only a few dozen shared terms.

- compared demo targets: {', '.join(selected)}
- alignment terms used: {proc_meta['n_alignment_terms']}

{md_table(rows5, ['term','cosine displacement','pre count','post count'])}

These demo displacement values have **no scientific interpretation**; they exist only to expose the output format and execution path.
"""
    (reports / "05_procrustes_alignment.md").write_text(report5, encoding="utf-8")

    jacc = pd.read_csv(d06 / "jaccard_overlap_common_vocab.csv")
    rows6 = [[r.term, int(r.k), round(float(r.jaccard), 4), int(r.overlap_count)] for r in jacc.itertuples()]
    report6 = f"""# 06. Nearest-neighbor and Jaccard outputs

The actual common-vocabulary nearest-neighbor/Jaccard implementation is used.

{md_table(rows6, ['term','k','Jaccard','overlap'])}

See `nearest_neighbor_table_common_vocab.csv` for the inspectable pre/post neighbor strings and `jaccard_overlap_common_vocab.csv` for set-overlap details.
"""
    (reports / "06_semantic_evidence.md").write_text(report6, encoding="utf-8")

    report7 = """# 07. Relation to the manuscript analysis

This directory demonstrates **file formats and data flow**, not manuscript-scale estimates.

The fixture differs intentionally in scale:

- synthetic text instead of copyrighted articles;
- relaxed phrase parameters so phrases can be visible in a tiny corpus;
- 30-dimensional demo Word2Vec models rather than the 200-dimensional manuscript models;
- `min_count=1` rather than 20;
- a 15-term alignment candidate set rather than 30,000/50,000 terms.

The repository's `examples/manuscript_output_snapshots/` directory contains derived, non-full-text outputs from the actual manuscript models: canonical displacement tables, alignment diagnostics, 30k-vs-50k sensitivity, nearest-neighbor/Jaccard outputs, and the 31-anchor consistency summary.

Canonical manuscript parameters are documented in the root `README.md` and preserved in `run_metadata/`.
"""
    (reports / "07_relation_to_manuscript.md").write_text(report7, encoding="utf-8")

    root_readme = """# Inspectable synthetic pipeline example

This directory is generated by `python examples/build_inspectable_examples.py`.
It provides a step-by-step, redistributable demonstration of the actual repository code path.
All source text is synthetic; no manuscript article text is included.

Read the reports in numerical order, then inspect the machine-readable files beside them.

```text
01_sources_and_extraction/
02_preprocessing/
03_shared_phrases/
04_word2vec/
05_procrustes/
06_semantic_evidence/
reports/
MANIFEST.json
```

The demonstration uses scaled-down modeling parameters because the fixture is tiny.
Do not use its numeric results as manuscript results. For actual derived manuscript outputs, see `../manuscript_output_snapshots/`.
"""
    (out / "README.md").write_text(root_readme, encoding="utf-8")

    # ------------------------------------------------------------------
    # File inventory with hashes. Exclude MANIFEST itself while constructing.
    # ------------------------------------------------------------------
    manifest = {
        "purpose": "Inspectable synthetic pipeline intermediates generated with actual repository code.",
        "synthetic_sources_only": True,
        "production_preprocessing_exercised": True,
        "demo_parameters_are_not_manuscript_parameters": True,
        "files": [],
    }
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name != "MANIFEST.json"):
        manifest["files"].append(
            {
                "path": p.relative_to(out).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
        )
    write_json(out / "MANIFEST.json", manifest)

    print("Inspectable-example export PASS")
    print(f"  output: {out}")
    print(f"  synthetic source records: {len(source_rows)}")
    print(f"  preprocessed units: pre={len(pre_prep)} post={len(post_prep)}")
    print(f"  shared vocabulary: {len(common)}")
    print(f"  demo targets: {', '.join(selected)}")
    print(f"  manifest files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
