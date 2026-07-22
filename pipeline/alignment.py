from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

from .config import PipelinePaths
from .example_outputs import write_displacement_summary
from .io_utils import write_tsv
from .reporting import write_stage_report


def load_vectors(path: str | Path) -> KeyedVectors:
    try:
        return Word2Vec.load(str(path)).wv
    except Exception:
        return KeyedVectors.load(str(path))


def get_count(vectors: KeyedVectors, token: str) -> int:
    try:
        return int(vectors.get_vecattr(token, "count"))
    except Exception:
        return 0


def is_clean_token(token: str) -> bool:
    return (
        len(token) >= 3
        and not bool(re.fullmatch(r"\d+", token))
        and bool(re.search(r"[A-Za-z]", token))
    )


def load_target_terms(config: dict[str, Any]) -> list[dict[str, str]]:
    analysis = config.get("analysis", {})
    if analysis.get("target_terms"):
        rows: list[dict[str, str]] = []
        for value in analysis["target_terms"]:
            if isinstance(value, str):
                rows.append({"term": value, "material_system": "target_terms"})
            else:
                rows.append(
                    {
                        "term": str(value["term"]),
                        "material_system": str(
                            value.get("material_system", value.get("group", "target_terms"))
                        ),
                    }
                )
        return list({(row["term"], row["material_system"]): row for row in rows}.values())

    path = Path(analysis["target_terms_path"])
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    import csv

    if path.suffix.lower() == ".txt":
        return [
            {"term": line.strip(), "material_system": "target_terms"}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = []
        for row in reader:
            term = row.get("term") or next(iter(row.values()))
            rows.append(
                {
                    "term": str(term).strip(),
                    "material_system": str(
                        row.get("material_system")
                        or row.get("group")
                        or row.get("category")
                        or "target_terms"
                    ).strip(),
                }
            )
        return rows


def build_alignment_terms(
    pre: KeyedVectors,
    post: KeyedVectors,
    target_terms: set[str],
    min_count: int,
    top_n: int,
) -> list[str]:
    candidates = []
    for token in set(pre.index_to_key) & set(post.index_to_key):
        if token in target_terms or not is_clean_token(token):
            continue
        c_pre, c_post = get_count(pre, token), get_count(post, token)
        if c_pre >= min_count and c_post >= min_count:
            candidates.append(token)
    candidates.sort(
        key=lambda token: (
            -min(get_count(pre, token), get_count(post, token)),
            token,
        )
    )
    return candidates[:top_n]


def fit_procrustes(
    pre: KeyedVectors,
    post: KeyedVectors,
    terms: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(terms) < 2:
        raise ValueError(
            "At least two alignment terms are required. Lower alignment.min_count "
            "or train on a larger corpus."
        )
    x_pre = normalize(np.vstack([pre[token] for token in terms]).astype(np.float64))
    x_post = normalize(np.vstack([post[token] for token in terms]).astype(np.float64))
    mu_pre = x_pre.mean(axis=0, keepdims=True)
    mu_post = x_post.mean(axis=0, keepdims=True)
    rotation, scale = orthogonal_procrustes(x_pre - mu_pre, x_post - mu_post)
    return rotation, mu_pre, mu_post, float(scale)


def aligned_pre_vector(
    vectors: KeyedVectors,
    token: str,
    rotation: np.ndarray,
    mu_pre: np.ndarray,
    mu_post: np.ndarray,
) -> np.ndarray:
    vector = normalize(vectors[token].reshape(1, -1).astype(np.float64))
    return ((vector - mu_pre) @ rotation + mu_post)[0]


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    periods = list(config["periods"])
    if len(periods) != 2:
        raise ValueError("Procrustes stage currently requires exactly two periods.")
    pre_period, post_period = periods
    pre = load_vectors(paths.model_path(pre_period))
    post = load_vectors(paths.model_path(post_period))
    if pre.vector_size != post.vector_size:
        raise ValueError("Pre and post model vector sizes differ.")

    target_rows = load_target_terms(config)
    target_terms = list(dict.fromkeys(row["term"] for row in target_rows))
    settings = config.get("alignment", {})
    alignment_terms = build_alignment_terms(
        pre,
        post,
        set(target_terms),
        min_count=int(settings.get("min_count", 100)),
        top_n=int(settings.get("top_n", 30_000)),
    )
    rotation, mu_pre, mu_post, scale = fit_procrustes(pre, post, alignment_terms)
    np.savez(
        paths.alignment_npz_path,
        R=rotation,
        mu_pre=mu_pre,
        mu_post=mu_post,
        scale=scale,
        pre_period=pre_period,
        post_period=post_period,
    )
    paths.alignment_terms_path.write_text(
        "".join(f"{term}\n" for term in alignment_terms),
        encoding="utf-8",
    )

    rows = []
    groups = {row["term"]: row["material_system"] for row in target_rows}
    for term in target_terms:
        in_pre, in_post = term in pre, term in post
        row = {
            "term": term,
            "material_system": groups.get(term, "target_terms"),
            "status": "compared" if in_pre and in_post else "missing",
            "in_pre": in_pre,
            "in_post": in_post,
            "pre_count": get_count(pre, term) if in_pre else 0,
            "post_count": get_count(post, term) if in_post else 0,
            "cosine_displacement": "",
        }
        if in_pre and in_post:
            aligned = aligned_pre_vector(pre, term, rotation, mu_pre, mu_post)
            post_vector = normalize(post[term].reshape(1, -1).astype(np.float64))[0]
            row["cosine_displacement"] = float(cosine(aligned, post_vector))
        rows.append(row)
    write_tsv(
        rows,
        paths.displacement_path,
        [
            "term",
            "material_system",
            "status",
            "in_pre",
            "in_post",
            "pre_count",
            "post_count",
            "cosine_displacement",
        ],
    )

    example_result = write_displacement_summary(config, paths)
    compared = [row for row in rows if row["status"] == "compared"]
    metrics = {
        "pre_period": pre_period,
        "post_period": post_period,
        "n_alignment_terms": len(alignment_terms),
        "n_target_terms": len(target_terms),
        "n_compared_terms": len(compared),
        "n_missing_terms": len(rows) - len(compared),
        "scale": scale,
        "orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))
        ),
    }
    write_stage_report(
        report_path=paths.reports_dir / "06_procrustes_alignment.md",
        title="Stage 6 — Orthogonal Procrustes alignment",
        purpose=(
            "Aligns the earlier-period embedding space to the later-period space "
            "using frequent shared background terms while excluding target terms."
        ),
        inputs=[paths.model_path(pre_period), paths.model_path(post_period)],
        outputs=[
            paths.alignment_npz_path,
            paths.alignment_terms_path,
            paths.displacement_path,
        ] + ([example_result[0]] if example_result else []),
        metrics=metrics,
        parameters=settings,
    )
    return metrics
