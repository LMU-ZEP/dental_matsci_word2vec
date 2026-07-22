# Stage 1 — Source manifest

Enumerates every source deterministically and records the exact subset selected for this run.

## Inputs


## Outputs

- `artifacts/01_manifest/source_manifest.jsonl`
- `../../examples/example_manifest.jsonl`

## Metrics

```json
{
  "n_manifest_records": 10,
  "n_selected": 10,
  "period_counts": {
    "pre2018": 5,
    "post2018": 5
  },
  "source_type_counts": {
    "pdf": 10
  }
}
```

## Parameters

```json
{
  "hash_files": true
}
```
