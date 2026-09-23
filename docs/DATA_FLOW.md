# Data flow and provenance map

This document maps every major manuscript stage to its actual code, inputs, outputs, and provenance status.

## 1. Source universe and selection

### PDF-derived records

**Inputs**
- bibliographic JSONL metadata;
- local PDF collection;
- optional period-specific Dental Materials Journal extra-PDF directory;
- XML source collection for cross-source exclusion during selection.

**Code**
- `scripts/corpus/get_valid_pdfs.py`

**Primary outputs**
- selected-title list;
- selected-record JSONL manifest;
- selection-statistics files.

**Provenance status**
- production selection manifests were preserved;
- historical selected counts: 45,926 pre-2018 PDFs and 32,207 post-2018 PDFs.

#### PDF period-assignment provenance

The preserved PDF-selection summary files contain historical selection-job parameters and should not by themselves be interpreted as definitions of the pre-2018/post-2018 corpus boundary. Period assignment of the final selected PDF records was therefore audited retrospectively.

| Period    | Selected PDF records | Record-level bibliographic year within period | Period verified from DMJ source-directory provenance | Known bibliographic year outside declared period |
| --------- | -------------------: | --------------------------------------------: | ---------------------------------------------------: | -----------------------------------------------: |
| pre-2018  |               45,926 |                                        43,489 |                                                2,437 |                                                0 |
| post-2018 |               32,207 |                                        31,069 |                                                1,138 |                                                0 |
| **Total** |           **78,133** |                                    **74,558** |                                            **3,575** |                                            **0** |

The `min_year` values preserved in the historical PDF-selection summary files were lower-bound parameters of already period-scoped source-selection jobs and should not be interpreted as the operational pre/post boundary. In particular, the preserved `min_year = 1997` value in the post-2018 selection summary does not indicate that records from 1997 onward were assigned to the post-2018 corpus.

For the metadata-backed selected PDFs, period assignment was verified retrospectively from record-level bibliographic year metadata. The remaining 3,575 records (2,437 pre-2018 and 1,138 post-2018) belong to the separately collected Dental Materials Journal subset for which record-level year was not retained in the stored selection records; their period assignment was verified from the corresponding period-specific source-directory provenance. No selected PDF with a known bibliographic year fell outside its declared corpus period.


### XML-derived records

**Inputs**
- Elsevier/ScienceDirect full-text XML files.

**Code**
- `scripts/corpus/get_xml_corpus.py`

**Primary output**
- raw JSON corpus of extracted text chunks.

**Provenance status**
- content was preserved in the raw corpus;
- source-record-level XML manifests were reconstructed retrospectively during the audit rather than preserved as production-run manifests;
- verified current source counts: 50,224 pre-2018 XML records and 36,988 post-2018 XML records.

The post-2018 XML content audit reproduced all 613,844 preserved raw XML items exactly as an order-independent multiset from the currently available 36,988 XML source files. The previously reported 36,998 source-record count could not be independently reproduced and is treated as a source-count typo/unresolved historical discrepancy, not as missing preserved corpus content.

---

## 2. Extraction, merge, and linguistic preprocessing

**Code**
- `scripts/corpus/pdf_extraction.py`
- `scripts/corpus/text_normalization.py`
- `scripts/corpus/word2vec_pipeline_step2_xml_pdf.py`

**Inputs per period**
- selected PDF records;
- XML raw corpus;
- period label and fixed seed.

**Active role of `word2vec_pipeline_step2_xml_pdf.py` in the manuscript workflow**

The current manuscript workflow uses this script to:

1. extract/merge selected PDF and XML raw text;
2. normalize scientific text;
3. preprocess with spaCy lemmatization and NLTK stopwords;
4. write the period-specific `*.preprocessed.json` token corpus.

The canonical manuscript workflow uses `pymupdf_columns` as the preferred PDF extraction backend, with `pypdf` as the fallback; `pypdf` remains available for explicit standalone use.

The script ends after writing the preprocessed corpus. The final manuscript phrase detector and Word2Vec models are built by the dedicated `scripts/modeling/` stages below. This separation is required because the final analysis uses one shared phrase detector for both periods.

**Canonical period outputs**

```text
outputs_1992_2017_xml_pdf/corpora/matsci_pre2018_seed42.preprocessed.json
outputs_after2018_xml_pdf/corpora/matsci_post2018_seed42.preprocessed.json
```

**Pre-shared-phrase token counts**

```text
pre-2018   254,001,830
post-2018  285,051,273
```

---

## 3. Shared phrase detection

**Code**
- `scripts/modeling/build_shared_phraser.py`

**Scientific requirement**

One phrase detector is trained on the combination of the two preprocessed corpora and then applied unchanged to each period. This avoids period-specific tokenization rules becoming an artificial source of temporal difference.

**Parameters used**

```text
phrase_min_count = 30
phrase_threshold = 10.0
max_vocab_size   = 40,000,000
```

**Outputs**

```text
outputs_shared/matsci_pre_post_shared_bigram.phraser
outputs_shared/matsci_pre2018_seed42.shared_phrased.json
outputs_shared/matsci_post2018_seed42.shared_phrased.json
outputs_shared/matsci_pre_post_shared_bigram.top_phrases.csv
outputs_shared/matsci_pre_post_shared_bigram.summary.json
```

**Final training-corpus token counts**

```text
pre-2018   234,184,910
post-2018  260,521,379
```

---

## 4. Word2Vec training

**Code**
- `scripts/modeling/train_word2vec_from_tokens.py`

**Inputs**
- the two period-specific shared-phrased corpora.

**Canonical parameters**

```text
skip-gram / sg=1
vector_size=200
window=8
min_count=20
epochs=15
sample=1e-4
negative=15
alpha=0.025
min_alpha=0.0005
seed=42
deterministic=True
workers=1
```

**Outputs**

```text
outputs_models/matsci_pre2018_sharedphrases.model
outputs_models/matsci_pre2018_sharedphrases.vocab.csv
outputs_models/matsci_pre2018_sharedphrases.config.json

outputs_models/matsci_post2018_sharedphrases.model
outputs_models/matsci_post2018_sharedphrases.vocab.csv
outputs_models/matsci_post2018_sharedphrases.config.json
```

**Vocabulary sizes**

```text
pre-2018   256,119
post-2018  278,189
```

Preserved snapshots of both training configurations are in `run_metadata/`.

---

## 5. Canonical target configuration

**File**
- `configs/target_terms.tsv`

It contains 56 material-system memberships representing 53 unique quantitative target terms. Three terms occur in two material systems:

```text
composite
fracture_toughness
translucency
```

A duplicated material-system membership is appropriate for group-level summaries, but each unique term contributes only once to global displacement and 30k-vs-50k sensitivity statistics.

`configs/visualization_terms.tsv` is a separate visualization configuration. It may include contextual or vocabulary-emergence terms that are not part of the canonical quantitative target set.

---

## 6. Primary orthogonal Procrustes alignment and displacement

**Code**
- `scripts/analysis/procrustes_displacement.py`

**Inputs**
- pre/post Word2Vec models;
- canonical target list;
- shared-background minimum count = 100;
- primary fitting vocabulary size = 30,000.

**Operation**

The pre-2018 embedding space is mapped to the post-2018 space by an orthogonal Procrustes transformation fitted on the 30,000 most frequent eligible shared background terms after excluding target terms.

For targets present in both vocabularies:

```text
cosine displacement = 1 - cosine(aligned pre vector, post vector)
```

**Primary outputs**

```text
outputs_procrustes_30000/procrustes_alignment.npz
outputs_procrustes_30000/alignment_terms.txt
outputs_procrustes_30000/cosine_displacement.csv
outputs_procrustes_30000/procrustes_metadata.json
```

### Historical metadata note

The preserved historical metadata in `run_metadata/procrustes_metadata_30000.json` records the older row-oriented target input. It produced 57 rows: 56 material-system memberships plus `everx_flow`, which was absent pre-2018 and therefore had no paired displacement. The canonical paired-analysis configuration now contains 53 unique terms. Since `everx_flow` was not in the shared pre/post vocabulary, removing it from the canonical target set does not change the eligible shared alignment background.

---

## 7. Alignment diagnostic

**Code**
- `scripts/analysis/procrustes_diagnostic.py`

**Input candidate set**
- the same 30,000 eligible shared background terms.

**Diagnostic split**

```text
24,000 fitting terms
 6,000 held-out evaluation terms
80/20 split, random_state=42
```

This diagnostic transformation is **separate from the primary 30k transformation** used to calculate target displacement.

**Metrics**
- same-word cosine distance before/after alignment;
- top-1/top-5/top-10 self-retrieval;
- orthogonality error.

**Canonical results**

```text
median distance before  0.997259
median distance after   0.269593
median per-term gain    0.724955
top-1                   0.8830
top-5                   0.9683
top-10                  0.9803
orthogonality error     5.03e-14
```

---

## 8. Alignment-vocabulary sensitivity

**Code**
- `scripts/analysis/procrustes_displacement.py` with `--top-n 50000`
- `scripts/analysis/calculate_procrustes_sensitivity.py`

The 50k alignment is fitted independently with the same scientific procedure. Sensitivity is then calculated across **53 unique paired target terms**, not repeated material-system memberships.

**Canonical results**

```text
Pearson r     0.9981916
Spearman rho  0.9928237
```

---

## 9. Reliability summaries

**Code**
- `scripts/analysis/calculate_procrustes_summary.py`

**Reliability criterion**

```text
min(pre_count, post_count) >= 1000
```

**Canonical global results**

```text
all unique targets                 n=53
mean displacement                  0.166066
median displacement                0.134409
SD                                 0.085485

reliability-filtered targets       n=43
mean displacement                  0.135829
median displacement                0.127491
SD                                 0.044156
```

The same script expands unique term values back through `configs/target_terms.tsv` for material-system summaries, where overlapping memberships are intentionally counted in each relevant group.

---

## 10. Nearest-neighbor and Jaccard analysis

**Code**
- `scripts/analysis/semantic_shift_evidence.py`

**Primary quantitative neighbor mode**
- common-vocabulary nearest neighbors;
- Jaccard overlap at `k=5, 10, 20`.

Restricting the primary comparison to the shared vocabulary distinguishes local neighborhood reorganization from simple post-period vocabulary emergence.

The same script also produces dimensionality-reduction coordinates/figures. PCA, t-SNE, and UMAP are qualitative visualization layers only; manuscript displacement is calculated in the aligned high-dimensional embedding space.

---

## 11. 90%-corpus consistency analysis

**Code**
- `scripts/robustness/sample90_and_train_word2vec.py`
- `scripts/robustness/compare_word2vec_consistency.py`

**Sampling**

```text
fraction: 0.90
pre seed: 42
post seed: 1042
```

The sampled models use the same Word2Vec hyperparameters as the full models.

**Anchor configuration**
- `configs/anchor_terms_consistency.txt` — 31 predefined anchors.

An anchor absent from either member of a full/sample pair is excluded from that period's quantitative metric.

**Canonical results**

```text
pre-2018:  27/31 anchors present in both
Jaccard@10 0.7471
Jaccard@20 0.7629
pairwise cosine-similarity Pearson 0.98314

post-2018: 31/31 anchors present in both
Jaccard@10 0.7938
Jaccard@20 0.7804
pairwise cosine-similarity Pearson 0.98367
```

---

## 12. Retrospective audits: separate evidence layer

These scripts verify provenance or revision-stage claims but do not sit on the model-training path.

### `audits/corpus/`
- XML source-manifest reconstruction;
- XML raw-corpus multiset comparison;
- PDF extraction quality comparisons;
- PDF year/DMJ provenance support;
- XML/PDF cross-source overlap checks;
- exact-file PDF duplicate checks.

### `audits/journals/`
- canonical journal normalization and journal-level source-record reconstruction;
- explicit alias and non-journal exclusion tables.

### `audits/terminology/`
- stage-wise 4Y/5Y notation audit;
- strict matches are separated from ambiguous spaced raw candidates and decimal-prefix false-positive candidates.

---

## 13. Validation path

The repository restructure was checked at three levels.

### Smoke test

```bash
python validation/run_smoke_test.py
```

Checks syntax, safe CLI entry points, canonical configuration sizes, and preserved model configs.

### Miniature end-to-end test

```bash
python validation/run_mini_e2e.py
```

Exercises:

```text
XML extraction
-> scientific text normalization
-> production spaCy/NLTK preprocessing when resources are available
-> shared phrase detection
-> deterministic pre/post Word2Vec
-> Procrustes + displacement
-> nearest-neighbor/Jaccard
```

The project-server run passed with the production preprocessing path enabled.

### Real-output regression test

```bash
python validation/run_regression_test.py
```

Uses preserved real outputs and verifies canonical downstream statistics without retraining the manuscript models.

All three tests passed on the project server.

---

## Inspectable intermediates distributed with the repository

Because the manuscript corpus contains copyrighted/subscription full text, source articles are not redistributed. Two separate reviewer-facing layers are provided instead.

### Synthetic end-to-end example

`examples/fixtures/` contains only synthetic XML/PDF text. Running:

```bash
python examples/build_inspectable_examples.py
```

uses the actual repository code to generate named files for source/extraction, production preprocessing, shared phrase detection, a tiny deterministic Word2Vec model, Procrustes displacement, and common-vocabulary neighbor/Jaccard analysis. Each stage also has a short Markdown report. Demo model/phrase parameters are intentionally scaled down for the tiny fixture and are explicitly marked as non-manuscript values.

### Actual manuscript-derived snapshots

`examples/manuscript_output_snapshots/` contains no article text but preserves compact actual downstream outputs: 53 unique target displacements, the 43-term reliability subset, global/group summaries, held-out alignment diagnostics, 30k-vs-50k sensitivity, common-vocabulary neighbors/Jaccard values, and the final 31-anchor consistency summary.

This separation lets reviewers inspect both **how the pipeline transforms data** and **the actual derived values underlying manuscript claims**, without representing synthetic demo results as manuscript evidence.
