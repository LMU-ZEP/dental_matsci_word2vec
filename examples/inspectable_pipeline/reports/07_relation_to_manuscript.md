# 07. Relation to the manuscript analysis

This directory demonstrates **file formats and data flow**, not manuscript-scale estimates.

The fixture differs intentionally in scale:

- synthetic text instead of copyrighted articles;
- relaxed phrase parameters so phrases can be visible in a tiny corpus;
- 30-dimensional demo Word2Vec models rather than the 200-dimensional manuscript models;
- `min_count=1` rather than 20;
- a 15-term alignment candidate set rather than 30,000/50,000 terms.

The repository's `examples/manuscript_output_snapshots/` directory contains derived, non-full-text outputs from the actual manuscript models: canonical displacement tables, alignment diagnostics, 30k-vs-50k sensitivity, nearest-neighbor/Jaccard outputs, and the 31-anchor consistency summary.

Canonical manuscript parameters are documented in the root `README.md` and preserved in `run_metadata/`.
