# Stage 7 — Semantic-shift evidence

Produces inspectable nearest-neighbor, neighborhood-overlap, and target-vocabulary-emergence tables.

## Inputs

- `artifacts/05_models/word2vec_pre2018.model`
- `artifacts/05_models/word2vec_post2018.model`
- `artifacts/06_alignment/procrustes_alignment.npz`

## Outputs

- `artifacts/07_evidence/neighbors.tsv`
- `artifacts/07_evidence/jaccard_overlap.tsv`
- `artifacts/07_evidence/vocabulary_emergence.tsv`
- `../../examples/example_neighbors.tsv`

## Metrics

```json
{
  "neighbor_mode": "common_vocab",
  "target_terms": 9,
  "compared_terms": 1,
  "neighbors_topn": 20,
  "jaccard_summary": {
    "5": {
      "n_terms": 1,
      "mean": 0.1111111111111111,
      "median": 0.1111111111111111
    },
    "10": {
      "n_terms": 1,
      "mean": 0.1111111111111111,
      "median": 0.1111111111111111
    },
    "20": {
      "n_terms": 1,
      "mean": 0.17647058823529413,
      "median": 0.17647058823529413
    }
  }
}
```

## Parameters

```json
{
  "neighbor_mode": "common_vocab",
  "neighbors_topn": 20,
  "k_values": [
    5,
    10,
    20
  ]
}
```
