from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from .alignment import get_count, load_target_terms, load_vectors
from .config import PipelinePaths
from .example_outputs import copy_neighbors
from .io_utils import write_tsv
from .reporting import write_stage_report


def _candidate_matrix(vectors, candidates: list[str]) -> tuple[list[str], np.ndarray]:
    present = [token for token in candidates if token in vectors]
    if not present:
        return [], np.empty((0, vectors.vector_size), dtype=np.float32)
    matrix = normalize(np.vstack([vectors[token] for token in present]).astype(np.float32))
    return present, matrix


def _neighbors(vectors, query: str, candidates: list[str], matrix: np.ndarray, topn: int):
    if query not in vectors or not candidates:
        return []
    query_vector = normalize(vectors[query].reshape(1, -1).astype(np.float32))[0]
    similarities = matrix @ query_vector
    if query in candidates:
        similarities[candidates.index(query)] = -np.inf
    indexes = np.argsort(similarities)[::-1][:topn]
    return [
        (candidates[index], float(similarities[index]))
        for index in indexes
        if np.isfinite(similarities[index])
    ]


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    periods = list(config["periods"])
    pre_period, post_period = periods
    pre = load_vectors(paths.model_path(pre_period))
    post = load_vectors(paths.model_path(post_period))
    target_rows = load_target_terms(config)
    settings = config.get("evidence", {})
    mode = settings.get("neighbor_mode", "common_vocab")
    topn = int(settings.get("neighbors_topn", 20))
    k_values = sorted({int(value) for value in settings.get("k_values", [5, 10, 20])})

    if mode == "common_vocab":
        common = sorted(set(pre.index_to_key) & set(post.index_to_key))
        pre_candidates = post_candidates = common
    elif mode == "full_vocab":
        pre_candidates = list(pre.index_to_key)
        post_candidates = list(post.index_to_key)
    else:
        raise ValueError("evidence.neighbor_mode must be common_vocab or full_vocab.")

    pre_terms, pre_matrix = _candidate_matrix(pre, pre_candidates)
    post_terms, post_matrix = _candidate_matrix(post, post_candidates)

    neighbor_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    emergence_rows: list[dict[str, Any]] = []
    compared = 0

    for target in target_rows:
        term = target["term"]
        group = target["material_system"]
        in_pre, in_post = term in pre, term in post
        if not in_pre and in_post:
            status = "post_only"
        elif in_pre and not in_post:
            status = "pre_only"
        elif in_pre and in_post:
            status = "present_in_both"
        else:
            status = "absent_in_both"
        emergence_rows.append(
            {
                "term": term,
                "material_system": group,
                "status": status,
                "pre_count": get_count(pre, term) if in_pre else 0,
                "post_count": get_count(post, term) if in_post else 0,
            }
        )

        max_neighbors = max([topn, *k_values])
        pre_neighbors = _neighbors(pre, term, pre_terms, pre_matrix, max_neighbors)
        post_neighbors = _neighbors(post, term, post_terms, post_matrix, max_neighbors)
        for period, values in ((pre_period, pre_neighbors), (post_period, post_neighbors)):
            if not values:
                neighbor_rows.append(
                    {
                        "term": term,
                        "material_system": group,
                        "period": period,
                        "rank": "",
                        "neighbor": "",
                        "similarity": "",
                        "status": "missing_query",
                        "mode": mode,
                    }
                )
            for rank, (neighbor, similarity) in enumerate(values[:topn], start=1):
                neighbor_rows.append(
                    {
                        "term": term,
                        "material_system": group,
                        "period": period,
                        "rank": rank,
                        "neighbor": neighbor,
                        "similarity": similarity,
                        "status": "compared",
                        "mode": mode,
                    }
                )

        if in_pre and in_post:
            compared += 1
            for k in k_values:
                pre_set = {token for token, _ in pre_neighbors[:k]}
                post_set = {token for token, _ in post_neighbors[:k]}
                union = pre_set | post_set
                intersection = pre_set & post_set
                jaccard_rows.append(
                    {
                        "term": term,
                        "material_system": group,
                        "mode": mode,
                        "k": k,
                        "overlap_count": len(intersection),
                        "jaccard": len(intersection) / len(union) if union else "",
                        "shared_neighbors": ";".join(sorted(intersection)),
                    }
                )

    write_tsv(
        neighbor_rows,
        paths.neighbors_path,
        [
            "term",
            "material_system",
            "period",
            "rank",
            "neighbor",
            "similarity",
            "status",
            "mode",
        ],
    )
    write_tsv(
        jaccard_rows,
        paths.jaccard_path,
        ["term", "material_system", "mode", "k", "overlap_count", "jaccard", "shared_neighbors"],
    )
    write_tsv(
        emergence_rows,
        paths.emergence_path,
        ["term", "material_system", "status", "pre_count", "post_count"],
    )

    example_path = copy_neighbors(config, paths)

    by_k: dict[int, list[float]] = defaultdict(list)
    for row in jaccard_rows:
        if row["jaccard"] != "":
            by_k[int(row["k"])].append(float(row["jaccard"]))
    metrics = {
        "neighbor_mode": mode,
        "target_terms": len(target_rows),
        "compared_terms": compared,
        "neighbors_topn": topn,
        "jaccard_summary": {
            str(k): {
                "n_terms": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for k, values in by_k.items()
        },
    }
    write_stage_report(
        report_path=paths.reports_dir / "07_semantic_evidence.md",
        title="Stage 7 — Semantic-shift evidence",
        purpose=(
            "Produces inspectable nearest-neighbor, neighborhood-overlap, and "
            "target-vocabulary-emergence tables."
        ),
        inputs=[
            paths.model_path(pre_period),
            paths.model_path(post_period),
            paths.alignment_npz_path,
        ],
        outputs=[paths.neighbors_path, paths.jaccard_path, paths.emergence_path]
        + ([example_path] if example_path else []),
        metrics=metrics,
        parameters=settings,
    )
    return metrics
