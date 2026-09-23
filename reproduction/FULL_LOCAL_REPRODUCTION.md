# Full local reproduction guide

## Scope

The public repository does not redistribute the complete article full-text corpus. Consequently there are two local starting points.

### Preferred deterministic starting point: period-specific preprocessed token corpora

If the following two preserved files are available locally:

```text
outputs_1992_2017_xml_pdf/corpora/matsci_pre2018_seed42.preprocessed.json
outputs_after2018_xml_pdf/corpora/matsci_post2018_seed42.preprocessed.json
```

then the shared phrase detector, both manuscript Word2Vec models, the primary and sensitivity Procrustes alignments, held-out diagnostic, reliability summaries, nearest-neighbor/Jaccard evidence, and 90%-corpus consistency analysis can be rerun with:

```bash
bash reproduction/run_from_preprocessed_corpora.sh
```

The script passes the manuscript parameters explicitly rather than relying on script defaults.

## Manuscript training settings

Shared phrase detector:

```text
min_count = 30
threshold = 10.0
max_vocab_size = 40,000,000
one detector trained on the combined pre/post corpora and applied unchanged to both
```

Word2Vec:

```text
architecture = skip-gram
vector_size = 200
window = 8
min_count = 20
epochs = 15
sample = 1e-4
negative = 15
alpha = 0.025
min_alpha = 0.0005
seed = 42
workers = 1
deterministic = true
```

Primary Procrustes:

```text
background min_count = 100
alignment terms = 30,000
pre -> post
canonical target set = 56 material-system memberships / 53 unique terms
```

Sensitivity alignment uses 50,000 background terms. The held-out diagnostic reproducibly splits the 30,000 eligible diagnostic background terms into 24,000 fitting and 6,000 evaluation terms (`random_state=42`).

## Expected corpus/model checkpoints

Preserved manuscript-run metadata records:

```text
pre shared-phrased tokens   234,184,910
post shared-phrased tokens  260,521,379
pre vocabulary              256,119
post vocabulary             278,189
```

The current verified source-record counts are:

```text
pre-2018:   96,150 = 50,224 XML + 45,926 PDF
post-2018:  69,195 = 36,988 XML + 32,207 PDF
```

The previously reported post XML count of 36,998 could not be independently reproduced from the currently available source files and is treated as a source-record-count reporting discrepancy, not as evidence that ten known training records can be recovered.

## Starting earlier: source selection and text extraction

The repository includes the actual corpus/extraction code under `scripts/corpus/` and retrospective audit tools under `audits/corpus/`. A complete source-level reconstruction additionally requires the locally held XML/PDF source directories and selection metadata. Those source documents are not part of the public package.

Important provenance distinction:

- production PDF selection/extraction was recorded in production-run manifests;
- XML source-record manifests used in the provenance audit were reconstructed retrospectively from the preserved XML sources using the original selection/extraction logic;
- therefore the repository does not claim that a public user can recreate the exact private source collection from repository files alone.

## Environment

Install dependencies and NLP resources:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download('stopwords')
PY
```

For the final archival bundle, `build_reproduction_bundle.py` records the installed package/version list, Python version, platform information, spaCy model version, and availability of NLTK stopwords.

## Verification after rerun

Use:

```bash
python3 validation/run_regression_test.py
```

for the bundled preserved-output regression test, and compare newly generated outputs with `reproduction/EXPECTED_RESULTS.json` and `examples/manuscript_output_snapshots/`.

A binary `.model` file hash is not the preferred cross-environment scientific equivalence criterion. The package records deterministic settings, but environment/library differences can affect serialized bytes. Scientific checkpoints are corpus counts, vocabulary counts, displacement values, diagnostic metrics, and consistency summaries.
