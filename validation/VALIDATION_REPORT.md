# Repository validation report

## Status

### Repository smoke test — PASS

- Python compilation passed.
- `--help` passed for all 12 executable pipeline/analysis scripts without starting a production run.
- User-specific hard-coded `/home/...` paths were removed from executable scripts.
- Canonical targets: 56 material-system memberships / 53 unique terms.
- Consistency anchors: 31 unique terms.
- Preserved pre/post Word2Vec run configurations match the manuscript parameters: vector size 200, window 8, min_count 20, epochs 15, sample 1e-4, negative 15, alpha 0.025, min_alpha 0.0005, skip-gram, seed 42, workers 1.

Local validation environment note: `ijson`, `en_core_web_sm`, and NLTK English stopwords are not installed in this container. This is an environment limitation, not a repository-source failure.

### Miniature end-to-end test — PASS

Passed:
- synthetic XML extraction;
- scientific text normalization;
- shared phrase detection trained once and applied to both periods;
- deterministic pre/post Word2Vec training;
- Procrustes alignment and target displacement;
- nearest-neighbor/Jaccard analysis.

Local container note: production spaCy/NLTK preprocessing was skipped locally because `en_core_web_sm` and NLTK stopwords were unavailable. The same test was then run on the project server and passed with `production preprocessing: PASS`, exercising the actual spaCy/NLTK preprocessing path.

### Regression test on preserved real outputs — PASS

No model retraining was performed or required.

Verified values:
- unique compared targets: 53;
- reliable targets: 43;
- 30k vs 50k Pearson r = 0.9981916329609319;
- 30k vs 50k Spearman rho = 0.992823738106757;
- held-out diagnostic: 24,000 fitting terms and 6,000 held-out terms;
- top-1 / top-5 / top-10 self-retrieval = 0.883 / 0.9683333333 / 0.9803333333;
- pre-2018 consistency: 27/31 anchors, Pearson = 0.9831416726;
- post-2018 consistency: 31/31 anchors, Pearson = 0.9836693947.

## Minimal source changes made during validation

No scientific algorithm was changed. Validation exposed three repository-entry-point issues, which were corrected:

1. `get_valid_pdfs.py` and `get_xml_corpus.py` no longer execute user-specific hard-coded production paths when run directly; both now expose normal command-line interfaces.
2. `calculate_procrustes_sensitivity.py` now exposes command-line input/output paths instead of reading fixed relative paths. Its unique-term sensitivity logic is unchanged.
3. Heavy runtime resources in the corpus-preprocessing script are initialized after argument parsing so `--help` is safe and side-effect free. Missing `ijson` now produces an explicit runtime dependency error when data processing is attempted rather than preventing CLI inspection.

These are execution/auditability changes only; the scientific computations used for the manuscript remain unchanged.

### Server portability fix
The mini-E2E fixture was revised after server validation exposed two test-only assumptions: a fixed corpus-unit count and an assumption that at least three predefined domain tokens would survive phrase detection. The current validation now checks corpus-unit preservation, uses a richer synthetic XML fixture, and selects three deterministic targets from the actual shared vocabulary while reserving at least three shared background terms for Procrustes fitting. These changes affect validation code only, not the production scientific pipeline.


## Project-server confirmation

On 2026-09-23, the inspectable examples were regenerated from the current repository source in the project-server environment using the documented production spaCy/NLTK resources.

The regenerated outputs passed the required inspectable-example validation:

```text
build_inspectable_examples.py  PASS
validate_examples.py           PASS
  synthetic fixtures           PASS
  manuscript-output snapshots  PASS
  generated inspectable pipeline PASS
```

The repository validation entry points were then rerun against the current source tree:

```text
run_smoke_test.py       PASS
run_mini_e2e.py         PASS (production spaCy/NLTK preprocessing exercised)
run_regression_test.py  PASS
```

This post-cleanup validation was performed after synchronization of the corpus preprocessing entry point, inspectable-example generation path, and canonical PDF extraction backend configuration.
