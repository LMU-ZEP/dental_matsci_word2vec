# Stage 2 — PDF/XML extraction

Extracts text from PDF and Elsevier XML sources into a shared JSONL schema while preserving source provenance.

## Inputs

- `artifacts/01_manifest/source_manifest.jsonl`

## Outputs

- `artifacts/02_extraction/raw_documents.jsonl`
- `artifacts/02_extraction/extraction_errors.jsonl`

## Metrics

```json
{
  "selected_documents": 10,
  "documents_with_text": 10,
  "chunks": 10,
  "chunks_pre2018": 5,
  "characters": 266439,
  "chunks_post2018": 5,
  "written_chunks": 10,
  "n_errors": 0
}
```

## Parameters

```json
{
  "xml_split": "section",
  "include_title": true,
  "include_abstract": true,
  "include_keywords": true,
  "include_section_titles": true,
  "min_chars": 30,
  "skip_unknown_year": false,
  "repair_pdf_after_eof": true
}
```

## Notes

- A single source document may produce multiple chunks (for example XML sections).
- Failures are retained in extraction_errors.jsonl rather than silently discarded.
