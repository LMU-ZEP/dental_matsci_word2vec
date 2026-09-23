# Reproducibility package

The reproduction bundle provides a fixed snapshot of the code, configuration, validation utilities, inspectable examples, and environment metadata corresponding to the submitted manuscript version. It is intended to preserve the exact reviewed/released repository state and to support inspection and the validation procedures documented below.

Reproducibility is described at four levels because the complete article full-text corpus cannot be redistributed.

The final reproduction bundle must be built from a clean Git checkout at the manuscript release tag.

## Reproducibility levels

### Level 1 — Public executable demonstration

Requires this repository the Python dependencies in `requirements.txt`, and the documented spaCy en_core_web_sm model and NLTK English stopwords resource.

It runs the real repository code on synthetic, redistributable XML/PDF fixtures and exports named intermediate artifacts for every major stage:

```bash
bash reproduction/run_public_demo.sh
```

Expected result:

```text
Inspectable-example export PASS
Inspectable-example validation PASS
```

This demonstrates the data flow and file formats. The tiny demonstration corpus and its numerical outputs are not manuscript results.

### Level 2 — Verification of manuscript-derived outputs

Does **not** retrain Word2Vec models. It checks the preserved manuscript-derived CSV/JSON snapshots and recomputes the canonical downstream summary statistics:

```bash
bash reproduction/run_verification_only.sh
```

This verifies, among other quantities, 53 unique target terms, 43 reliability-filtered terms, the corrected 30k-vs-50k sensitivity statistics, held-out Procrustes diagnostics, and 31-anchor consistency summaries.

### Level 3 — Reanalysis from preserved trained models

This level requires the two full manuscript Word2Vec models, which are not distributed in this repository. Given these models, the primary and sensitivity Procrustes alignments, target-term displacement analyses, held-out alignment diagnostics, reliability summaries, and nearest-neighbor/Jaccard analyses can be rerun using the commands documented in `FULL_LOCAL_REPRODUCTION.md` and the root `README.md`.

Reproducing the full-corpus-versus-90%-corpus consistency analysis additionally requires either the corresponding pre-trained 90%-corpus models or the period-specific shared-phrased corpora from which the 90%-corpus models can be retrained.

No retraining of the full Word2Vec models is required at this level.

### Level 4 — Full local training reconstruction

Requires the non-redistributable period-specific text/token corpora or the original locally held source files. From the preserved **preprocessed period-specific JSON token corpora**, the shared phrase detector, two Word2Vec models, and all downstream analyses can be rebuilt with:

```bash
bash reproduction/run_from_preprocessed_corpora.sh
```

Reconstructing the corpora themselves from original source articles additionally requires the local source directories and source-selection information described in `FULL_LOCAL_REPRODUCTION.md`. Those full texts are not included because of copyright/access restrictions.

## Build a release bundle

After `examples/inspectable_pipeline/` has been generated and all validation tests pass, build a sanitized, checksummed bundle with:

```bash
python3 reproduction/build_reproduction_bundle.py
```

By default this:

1. runs the repository validation suite;
2. requires the generated inspectable example;
3. records Python/platform/package information;
4. copies the repository while excluding caches, `.git`, `dist`, and private/full-corpus output directories;
5. writes `FILE_MANIFEST.csv`, `SHA256SUMS`, and `BUILD_INFO.json`;
6. creates `dist/dental_matsci_word2vec_reproducibility_package.zip`.

The resulting zip is the artifact to archive together with the final tagged repository release.

## Important distinction

`examples/inspectable_pipeline/` is synthetic demonstration output. `examples/manuscript_output_snapshots/` contains compact **real derived outputs** from the manuscript analyses but no full article text. `run_metadata/` contains preserved run settings and summary metadata. Retrospective provenance audits under `audits/` are evidence about the corpus and terminology; they are not represented as production training steps.
