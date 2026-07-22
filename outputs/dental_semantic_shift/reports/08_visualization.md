# Stage 8 — Visualization

Creates optional projections from Procrustes-aligned target vectors.

## Inputs

- `artifacts/06_alignment/procrustes_alignment.npz`

## Outputs

- `artifacts/08_visualization`
- `artifacts/08_visualization/visualization_metadata.json`

## Metrics

```json
{
  "enabled": true,
  "plots_written": 1,
  "sets": {
    "structural_color_composite": {
      "status": "written",
      "n_points": 2,
      "methods": [
        "pca"
      ]
    }
  }
}
```

## Parameters

```json
{
  "enabled": true,
  "plot_sets": {
    "structural_color_composite": [
      "opacity",
      "refractive_index",
      "composite"
    ]
  }
}
```
