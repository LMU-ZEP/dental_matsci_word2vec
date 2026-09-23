# Inspectable examples and manuscript-output snapshots

This directory has two deliberately separate purposes.

## `fixtures/` and `inspectable_pipeline/`

`fixtures/` contains only synthetic, redistributable XML/PDF source files. They are not articles from the manuscript corpus.

Run:

```bash
python examples/build_inspectable_examples.py
```

The command exercises the repository's actual extraction, production spaCy/NLTK preprocessing, shared phrase-detection, deterministic Word2Vec, Procrustes, and nearest-neighbor/Jaccard code, then writes named intermediates to `inspectable_pipeline/`.

The demonstration uses scaled-down modeling parameters because the fixture is tiny. The resulting numeric values are not manuscript results and must not be interpreted scientifically.

## `manuscript_output_snapshots/`

This folder contains compact **derived outputs from the actual manuscript models**. It contains no article full text. These snapshots let a reviewer inspect the final quantitative tables that underlie manuscript claims without redistributing copyrighted source material.

The two layers should not be mixed:

- synthetic pipeline example = inspect file formats and stage-to-stage transformations;
- manuscript output snapshots = inspect actual derived analysis values.
