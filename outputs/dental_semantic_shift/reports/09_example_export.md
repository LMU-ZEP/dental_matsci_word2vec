# Stage 9 — Example artifact verification

Verifies that every preceding stage wrote its reviewer-facing example artifact with the requested filename.

## Inputs

- `../../examples/example_config.json`
- `../../examples/example_manifest.jsonl`
- `../../examples/example_preprocessed_tokens.txt`
- `../../examples/example_phrased_tokens.txt`
- `../../examples/example_vocabulary_counts.tsv`
- `../../examples/example_neighbors.tsv`
- `../../examples/example_displacement_summary.tsv`

## Outputs

- `../../examples/example_config.json`
- `../../examples/example_manifest.jsonl`
- `../../examples/example_preprocessed_tokens.txt`
- `../../examples/example_phrased_tokens.txt`
- `../../examples/example_vocabulary_counts.tsv`
- `../../examples/example_neighbors.tsv`
- `../../examples/example_displacement_summary.tsv`

## Metrics

```json
{
  "enabled": true,
  "n_expected_files": 7,
  "n_existing_files": 7,
  "files": {
    "config": "example_config.json",
    "manifest": "example_manifest.jsonl",
    "preprocessed": "example_preprocessed_tokens.txt",
    "phrased": "example_phrased_tokens.txt",
    "vocabulary": "example_vocabulary_counts.tsv",
    "neighbors": "example_neighbors.tsv",
    "displacement_summary": "example_displacement_summary.tsv"
  }
}
```

## Parameters

```json
{
  "enabled": true,
  "output_dir": "../../examples"
}
```

## Notes

- The files are produced incrementally: manifest, preprocessing, phrases, training, alignment, and evidence each write their own example artifact.
