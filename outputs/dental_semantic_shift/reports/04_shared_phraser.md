# Stage 4 — Shared phrase detection

Fits one shared gensim Phraser on all periods and applies the same transformation to every period before model training.

## Inputs

- `artifacts/03_preprocessing/preprocessed_tokens.jsonl`

## Outputs

- `artifacts/04_phrases/shared_phraser.phraser`
- `artifacts/04_phrases/phrased_tokens.jsonl`
- `artifacts/04_phrases/phrase_counts.tsv`
- `../../examples/example_phrased_tokens.txt`

## Metrics

```json
{
  "documents": 10,
  "input_tokens": 26214,
  "output_tokens": 25967,
  "phrase_tokens": 247,
  "unique_phrase_tokens": 4
}
```

## Parameters

```json
{
  "min_count": 50,
  "threshold": 10.0,
  "max_vocab_size": 40000000,
  "progress_every": 10000,
  "delimiter": "_"
}
```
