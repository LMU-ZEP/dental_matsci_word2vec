# 03. Shared phrase detection

The actual shared-phraser implementation is used, but the tiny demonstration requires relaxed parameters:

- demo `min_count = 2`
- demo `threshold = 0.1`

These are **not** the manuscript values (`min_count=30`, `threshold=10.0`). The scientific requirement is the same: one phraser is trained on the combined pre/post corpora and applied unchanged to both periods.

Top phrase tokens in this demo:

| rank | phrase | combined count |
| --- | --- | --- |
| 1 | matching_color | 16 |
| 2 | fracture_toughness | 16 |
| 3 | shade_matching | 16 |
| 4 | color_matching | 16 |
| 5 | zirconia_fracture | 12 |
| 6 | strength_surface | 12 |
| 7 | ceramic_testing | 12 |
| 8 | clinical_restoration | 12 |
| 9 | filler_matrix | 12 |
| 10 | wear_hardness | 12 |

