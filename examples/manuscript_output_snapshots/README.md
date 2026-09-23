# Derived manuscript-output snapshots

These files are compact, non-full-text outputs from the actual manuscript analysis. They are included so a reviewer can inspect the quantitative inputs behind reported results without access to copyrighted article text or the full trained models.

Neighbor snapshots are preserved exactly as generated and have not been manually cleaned post hoc; occasional extraction/tokenization artifacts may therefore remain visible. Such artifacts were not selected as target terms or used for qualitative interpretation.

Contents:

- `cosine_displacement_unique_terms.csv` — 53 unique paired target terms from the primary 30k alignment;
- `cosine_displacement_reliable_unique_terms.csv` — the 43 terms meeting `min(pre_count, post_count) >= 1000`;
- `cosine_displacement_global_summary.csv` — global all/unique/reliability summaries;
- `cosine_displacement_group_summary.csv` — material-system summaries;
- `alignment_diagnostic_summary.csv` — 24,000-fit / 6,000-held-out Procrustes diagnostic;
- `procrustes_30k_50k_sensitivity.csv` — unique-term 30k-vs-50k displacement comparison;
- `nearest_neighbor_table_common_vocab.csv` — actual pre/post common-vocabulary neighbors for the 53 canonical terms;
- `jaccard_overlap_common_vocab.csv` — actual k=5/10/20 neighbor-set overlaps;
- `consistency_summary_31anchors.json` — full-vs-90%-corpus consistency summary for the final 31-anchor set.

These files are derived outputs, not a substitute for the private source corpus. Canonical parameters and provenance notes are documented in the root README, `docs/DATA_FLOW.md`, and `run_metadata/`.
