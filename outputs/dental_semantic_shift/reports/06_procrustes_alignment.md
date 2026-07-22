# Stage 6 — Orthogonal Procrustes alignment

Aligns the earlier-period embedding space to the later-period space using frequent shared background terms while excluding target terms.

## Inputs

- `artifacts/05_models/word2vec_pre2018.model`
- `artifacts/05_models/word2vec_post2018.model`

## Outputs

- `artifacts/06_alignment/procrustes_alignment.npz`
- `artifacts/06_alignment/alignment_terms.txt`
- `artifacts/06_alignment/cosine_displacement.tsv`
- `../../examples/example_displacement_summary.tsv`

## Metrics

```json
{
  "pre_period": "pre2018",
  "post_period": "post2018",
  "n_alignment_terms": 449,
  "n_target_terms": 9,
  "n_compared_terms": 1,
  "n_missing_terms": 8,
  "scale": 0.16521585288769935,
  "orthogonality_error": 4.792987195326319e-14
}
```

## Parameters

```json
{
  "min_count": 2,
  "top_n": 30000
}
```
