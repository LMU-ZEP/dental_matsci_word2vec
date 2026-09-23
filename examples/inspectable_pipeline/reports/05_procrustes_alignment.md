# 05. Procrustes alignment and displacement

The actual `procrustes_displacement.py` is run on the two tiny demonstration models.
The alignment vocabulary is deliberately small (`top_n=15`, `min_count=1`) because the fixture contains only a few dozen shared terms.

- compared demo targets: fracture_toughness, shade_matching, color_matching
- alignment terms used: 15

| term | cosine displacement | pre count | post count |
| --- | --- | --- | --- |
| fracture_toughness | 0.078662 | 8 | 8 |
| shade_matching | 0.099158 | 8 | 8 |
| color_matching | 0.097029 | 8 | 8 |


These demo displacement values have **no scientific interpretation**; they exist only to expose the output format and execution path.
