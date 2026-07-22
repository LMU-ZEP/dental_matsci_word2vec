# Stage 5 — Period-specific Word2Vec training

Trains one model per period from corpora transformed by the same shared phraser.

## Inputs

- `artifacts/04_phrases/phrased_tokens.jsonl`
- `artifacts/04_phrases/shared_phraser.phraser`

## Outputs

- `artifacts/05_models/vocabulary_counts.tsv`
- `artifacts/05_models/word2vec_pre2018.model`
- `artifacts/05_models/word2vec_pre2018.config.json`
- `artifacts/05_models/word2vec_post2018.model`
- `artifacts/05_models/word2vec_post2018.config.json`
- `../../examples/example_vocabulary_counts.tsv`

## Metrics

```json
{
  "periods": {
    "pre2018": {
      "n_documents": 5,
      "n_tokens": 9152,
      "empty_documents": 0,
      "max_document_tokens": 2307,
      "avg_tokens_per_document": 1830.4,
      "vocabulary_size": 1015,
      "model_path": "artifacts/05_models/word2vec_pre2018.model"
    },
    "post2018": {
      "n_documents": 5,
      "n_tokens": 16815,
      "empty_documents": 0,
      "max_document_tokens": 3996,
      "avg_tokens_per_document": 3363.0,
      "vocabulary_size": 1800,
      "model_path": "artifacts/05_models/word2vec_post2018.model"
    }
  },
  "workers_effective": 1,
  "seed": 42
}
```

## Parameters

```json
{
  "vector_size": 200,
  "window": 8,
  "min_count": 2,
  "epochs": 10,
  "sample": 0.0001,
  "alpha": 0.025,
  "min_alpha": 0.0005,
  "negative": 10,
  "skip_gram": false,
  "deterministic": true,
  "progress_every": 10000
}
```

## Notes

- deterministic=true forces workers=1; this is required for reproducible gensim training.
