from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from .alignment import aligned_pre_vector, get_count, load_target_terms, load_vectors
from .config import PipelinePaths
from .reporting import write_stage_report


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    settings = config.get("visualization", {})
    if not bool(settings.get("enabled", False)):
        metrics = {"enabled": False, "plots_written": 0}
        write_stage_report(
            report_path=paths.reports_dir / "08_visualization.md",
            title="Stage 8 — Visualization",
            purpose="Optional PCA, t-SNE, and UMAP projections of aligned target vectors.",
            inputs=[paths.alignment_npz_path],
            outputs=[],
            metrics=metrics,
            parameters=settings,
            notes=["This stage was disabled in the configuration."],
        )
        return metrics

    import matplotlib.pyplot as plt

    periods = list(config["periods"])
    pre_period, post_period = periods
    pre = load_vectors(paths.model_path(pre_period))
    post = load_vectors(paths.model_path(post_period))
    alignment = np.load(paths.alignment_npz_path)
    rotation, mu_pre, mu_post = alignment["R"], alignment["mu_pre"], alignment["mu_post"]
    term_rows = load_target_terms(config)
    plot_sets = settings.get(
        "plot_sets",
        {"all_targets": [row["term"] for row in term_rows]},
    )

    plots_written = 0
    metadata: dict[str, Any] = {}
    paths.visualization_dir.mkdir(parents=True, exist_ok=True)

    for set_name, selected_terms in plot_sets.items():
        labels: list[tuple[str, str]] = []
        vectors: list[np.ndarray] = []
        for term in selected_terms:
            if term in pre and term in post:
                labels.extend([(term, pre_period), (term, post_period)])
                vectors.extend(
                    [
                        aligned_pre_vector(pre, term, rotation, mu_pre, mu_post),
                        normalize(post[term].reshape(1, -1).astype(np.float64))[0],
                    ]
                )
            elif term in post:
                labels.append((term, f"{post_period}_only"))
                vectors.append(normalize(post[term].reshape(1, -1).astype(np.float64))[0])

        if len(vectors) < 2:
            metadata[set_name] = {"status": "skipped", "n_points": len(vectors)}
            continue
        matrix = np.vstack(vectors)
        methods: dict[str, np.ndarray] = {"pca": PCA(n_components=2).fit_transform(matrix)}
        if len(vectors) >= 6:
            perplexity = min(30, max(2, (len(vectors) - 1) // 3))
            methods["tsne"] = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=int(config.get("seed", 42)),
            ).fit_transform(matrix)
        try:
            import umap

            if len(vectors) >= 4:
                methods["umap"] = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, max(2, len(vectors) - 1)),
                    metric="cosine",
                    random_state=int(config.get("seed", 42)),
                ).fit_transform(matrix)
        except ImportError:
            pass

        for method, coords in methods.items():
            output = paths.visualization_dir / f"{set_name}_{method}.png"
            plt.figure(figsize=(10, 7))
            plt.scatter(coords[:, 0], coords[:, 1])
            for (term, period), (x, y) in zip(labels, coords):
                plt.text(x, y, f"{term} [{period}]", fontsize=8)
            plt.title(f"{set_name}: {method.upper()} projection")
            plt.tight_layout()
            plt.savefig(output, dpi=300)
            plt.close()
            plots_written += 1
        metadata[set_name] = {"status": "written", "n_points": len(vectors), "methods": list(methods)}

    metadata_path = paths.visualization_dir / "visualization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = {"enabled": True, "plots_written": plots_written, "sets": metadata}
    write_stage_report(
        report_path=paths.reports_dir / "08_visualization.md",
        title="Stage 8 — Visualization",
        purpose="Creates optional projections from Procrustes-aligned target vectors.",
        inputs=[paths.alignment_npz_path],
        outputs=[paths.visualization_dir, metadata_path],
        metrics=metrics,
        parameters=settings,
    )
    return metrics
