# Modular dental Word2Vec pipeline

This repository reorganizes the original scripts into one `main.py` orchestrator and a separate module for every inspectable stage:

1. source manifest;
2. PDF/XML extraction;
3. preprocessing;
4. shared phrase detection;
5. period-specific Word2Vec training;
6. Orthogonal Procrustes alignment;
7. semantic-shift evidence;
8. optional visualization;
9. reviewer-facing example export.

Every stage writes both machine-readable artifacts and a named Markdown report under `RUN_DIR/reports/`.

## Project layout

```text
main.py
pipeline/
  manifest.py
  pdf_extraction.py
  xml_extraction.py
  extraction.py
  preprocessing.py
  phrases.py
  word2vec_training.py
  alignment.py
  evidence.py
  visualization.py
  example_export.py
configs/
  pipeline_config.example.json
  target_terms.tsv
examples/
  example_config.json
  example_manifest.jsonl
  example_preprocessed_tokens.txt
  example_phrased_tokens.txt
  example_vocabulary_counts.tsv
  example_neighbors.tsv
  example_displacement_summary.tsv
  reports/
    01_manifest.md
    ...
    09_example_export.md
```

## Run the bundled example

```bash
python -m pip install -r requirements.txt
python main.py --config examples/example_config.json
```

The example uses the lightweight `simple` preprocessing engine so that it runs without downloading a spaCy language model. It still exercises the shared phraser, two Word2Vec models, Procrustes alignment, nearest neighbors, and example export.

## Run the production pipeline

1. Copy `configs/pipeline_config.example.json` and replace the source paths.
2. Put target terms in a TSV with columns `term` and `material_system`.
3. Install the production spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. Run:

```bash
python main.py --config configs/my_pipeline_config.json
```

## Resume or rerun selected stages

Artifacts have stable names, so stages can be rerun independently after their dependencies exist:

```bash
python main.py --config configs/my_pipeline_config.json \
  --stages alignment evidence visualization
```

## Main artifact schema

All document-level intermediate files use JSONL. Each record retains `document_id`, `chunk_id`, `period`, `source_type`, `source_path`, publication year, and either raw text or tokens. TSV outputs are deliberately long-form.

## Reproducibility

- source files are sorted before selection;
- the manifest records the selected subset and optional SHA-256 hashes;
- one shared `Phraser` is fitted across both periods;
- target terms are excluded from the alignment vocabulary;
- `word2vec.deterministic=true` forces `workers=1`;
- the resolved configuration and a report for every stage are saved with the run.
