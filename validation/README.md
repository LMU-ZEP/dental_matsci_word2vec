# Repository validation

This directory contains three independent validation checks for the repository structure and scientific analysis code.

## 1. Repository smoke test

```bash
python validation/run_smoke_test.py
```

Checks Python syntax, side-effect-free `--help` for the public CLI scripts, absence of user-specific `/home/...` paths in executable Python code, canonical target/anchor counts, and agreement of the preserved pre/post Word2Vec run configurations with the manuscript parameters.

The environment line is informational. Missing runtime resources such as `ijson`, `en_core_web_sm`, or NLTK stopwords should be installed before a full production run; they do not make the source tree itself invalid.

## 2. Miniature end-to-end test

```bash
python validation/run_mini_e2e.py
```

Uses synthetic redistributable data and exercises:

XML extraction -> text normalization -> shared phrase detection -> deterministic Word2Vec -> Procrustes alignment/displacement -> nearest-neighbor/Jaccard analysis.

If `en_core_web_sm` and NLTK English stopwords are available, the test also executes the production preprocessing function on the synthetic raw corpus. If they are unavailable, that sub-step is reported as `SKIPPED`; all downstream stages still run using deterministic synthetic token corpora.

If the real `ijson` package is unavailable, only the miniature test uses a tiny compatibility shim under `validation/compat/`. The full corpus must use the real streaming `ijson` dependency.

## 3. Regression test against preserved real outputs

```bash
python validation/run_regression_test.py
```

This test **does not retrain Word2Vec models**. It uses preserved outputs from the real analysis and checks that the current repository code reproduces the reported downstream statistics, including:

- 53 unique compared target terms;
- 43 terms meeting `min(pre_count, post_count) >= 1000`;
- 30k-vs-50k sensitivity: Pearson 0.9981916 and Spearman 0.9928237;
- held-out Procrustes diagnostic: 24,000 fitting / 6,000 held-out terms, top-1 accuracy 0.883;
- 90%-corpus consistency: 27/31 pre-2018 anchors and 31/31 post-2018 anchors.

Model retraining belongs to a later clean-room/full-reproduction test. It is intentionally not required for this regression check because the purpose here is to detect changes in downstream analysis code given fixed scientific inputs.

## Mini E2E design note

The miniature test uses several small synthetic XML records with a deliberately rich shared vocabulary. When the production spaCy/NLTK resources are available, the XML-derived raw corpus is passed through the production preprocessing script before phrase detection and downstream analysis. Target terms for the miniature Procrustes check are selected deterministically from the actual shared vocabulary after phrase detection, so the test does not depend on a particular gensim phrase segmentation.
