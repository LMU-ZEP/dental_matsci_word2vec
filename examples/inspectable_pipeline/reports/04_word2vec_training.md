# 04. Word2Vec training

The actual deterministic Word2Vec training script is used. Parameters are scaled down only for this tiny fixture.

```json
{
  "vector_size": 30,
  "window": 5,
  "min_count": 1,
  "epochs": 20,
  "sample": 0.001,
  "negative": 5,
  "alpha": 0.025,
  "min_alpha": 0.0005,
  "skip_gram": true,
  "seed": 42,
  "workers": 1
}
```

Vocabulary sizes:

- pre demo model: 63
- post demo model: 71

`trained_vectors_sample.csv` exposes the first eight coordinates for 12 shared terms. Full tiny Gensim model files are also retained for exact downstream inspection.
