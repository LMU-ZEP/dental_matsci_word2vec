# Reproducibility matrix

| Claim / stage | Publicly runnable from this repository? | Requires private/full-text data? | Primary evidence |
|---|---:|---:|---|
| XML/PDF extraction and text cleanup | Yes, on synthetic fixtures | Yes for the complete article corpus | `examples/inspectable_pipeline/01_sources_and_extraction/`; `scripts/corpus/` |
| Production preprocessing | Yes, on synthetic fixtures | Yes for the complete corpus | `examples/inspectable_pipeline/02_preprocessing/`; `validation/VALIDATION_REPORT.md` |
| Shared phrase detection | Yes, on synthetic fixtures | Yes for full-model reconstruction | `scripts/modeling/build_shared_phraser.py`; preserved phraser summary |
| Word2Vec training mechanics | Yes, on synthetic fixtures | Yes for manuscript models | `scripts/modeling/train_word2vec_from_tokens.py`; preserved model configs |
| Manuscript Word2Vec hyperparameters | Yes, inspectable | No | `run_metadata/matsci_*sharedphrases.config.json` |
| Primary 30k Procrustes method | Yes, on synthetic fixtures | Trained manuscript models for exact manuscript values | `scripts/analysis/procrustes_displacement.py` |
| 50k sensitivity | Exact derived values verifiable without retraining | No for verification; models for reanalysis | `examples/manuscript_output_snapshots/procrustes_30k_50k_sensitivity.csv` |
| Held-out 24k/6k diagnostic | Exact derived summary verifiable | Models for complete rerun | `alignment_diagnostic_summary.csv`; `procrustes_diagnostic.py` |
| Reliability filter | Exact derived values verifiable | No | canonical displacement snapshots + `calculate_procrustes_summary.py` |
| Nearest-neighbor/Jaccard evidence | Real derived output inspectable | Models for complete rerun | manuscript-output snapshots + `semantic_shift_evidence.py` |
| 90% consistency statistics | Exact summary verifiable | Full/90% models for complete rerun | `consistency_summary_31anchors.json` |
| 4Y/5Y notation audit | Audit code and preserved assessment inspectable | Full raw/preprocessed corpora for complete rerun | `audits/terminology/`; `run_metadata/4y5y_claim_assessment.txt` |
| Journal/source provenance audits | Audit code inspectable | Original local manifests/source directories for complete rerun | `audits/corpus/`, `audits/journals/` |

The package therefore distinguishes **public executable demonstration**, **verification of preserved manuscript-derived outputs**, **model-level reanalysis**, and **full local training reconstruction**. These are not described as equivalent levels of reproducibility.
