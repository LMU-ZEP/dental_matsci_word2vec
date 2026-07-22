# Stage 3 — Text preprocessing

Normalizes extracted text and writes one inspectable token list per JSONL record.

## Inputs

- `artifacts/02_extraction/raw_documents.jsonl`

## Outputs

- `artifacts/03_preprocessing/preprocessed_tokens.jsonl`
- `../../examples/example_preprocessed_tokens.txt`

## Metrics

```json
{
  "documents": 10,
  "tokens": 26214,
  "documents_pre2018": 5,
  "tokens_pre2018": 9219,
  "documents_post2018": 5,
  "tokens_post2018": 16995,
  "written_documents": 10,
  "engine": "spacy",
  "avg_tokens_per_document": 2621.4
}
```

## Parameters

```json
{
  "engine": "spacy",
  "spacy_model": "en_core_web_sm",
  "disable": [
    "ner",
    "parser"
  ],
  "max_length": 3000000,
  "batch_size": 32,
  "lowercase": true,
  "lemmatize": true,
  "remove_stopwords": true,
  "stopword_source": "nltk",
  "remove_numbers": true
}
```

## Notes

- Use engine='spacy' for the production lemmatized corpus.
- The lightweight engine='simple' is intended for the bundled example and CI smoke tests.
