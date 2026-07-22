from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelinePaths
from .io_utils import iter_jsonl, read_tsv, write_tsv


EXAMPLE_FILENAMES = {
    "config": "example_config.json",
    "manifest": "example_manifest.jsonl",
    "preprocessed": "example_preprocessed_tokens.txt",
    "phrased": "example_phrased_tokens.txt",
    "vocabulary": "example_vocabulary_counts.tsv",
    "neighbors": "example_neighbors.tsv",
    "displacement_summary": "example_displacement_summary.tsv",
}


def enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("example_export", {}).get("enabled", False))


def output_dir(config: dict[str, Any], paths: PipelinePaths) -> Path:
    value = config.get("example_export", {}).get(
        "output_dir", paths.project_root / "examples"
    )
    directory = Path(value)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_path(
    config: dict[str, Any],
    paths: PipelinePaths,
    key: str,
) -> Path:
    return output_dir(config, paths) / EXAMPLE_FILENAMES[key]


def copy_config(
    config_path: Path,
    config: dict[str, Any],
    paths: PipelinePaths,
) -> Path | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "config")
    if config_path.resolve() != destination.resolve():
        shutil.copyfile(config_path, destination)
    return destination


def copy_manifest(config: dict[str, Any], paths: PipelinePaths) -> Path | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "manifest")
    shutil.copyfile(paths.manifest_path, destination)
    return destination


def _write_tokens_txt(input_path: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_jsonl(input_path):
            handle.write(
                f"{record['chunk_id']}\t{record['period']}\t"
                + " ".join(record.get("tokens", []))
                + "\n"
            )
            count += 1
    return count


def write_preprocessed(config: dict[str, Any], paths: PipelinePaths) -> tuple[Path, int] | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "preprocessed")
    return destination, _write_tokens_txt(paths.preprocessed_path, destination)


def write_phrased(config: dict[str, Any], paths: PipelinePaths) -> tuple[Path, int] | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "phrased")
    return destination, _write_tokens_txt(paths.phrased_path, destination)


def copy_vocabulary(config: dict[str, Any], paths: PipelinePaths) -> Path | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "vocabulary")
    shutil.copyfile(paths.vocabulary_counts_path, destination)
    return destination


def copy_neighbors(config: dict[str, Any], paths: PipelinePaths) -> Path | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "neighbors")
    shutil.copyfile(paths.neighbors_path, destination)
    return destination


def _displacement_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("cosine_displacement", "")
        if row.get("status") == "compared" and value not in {"", None}:
            displacement = float(value)
            grouped["all"].append(displacement)
            grouped[row.get("material_system", "target_terms")].append(displacement)

    summary = []
    for scope, values in grouped.items():
        summary.append(
            {
                "scope": scope,
                "n_terms": len(values),
                "mean_cosine_displacement": float(np.mean(values)),
                "median_cosine_displacement": float(np.median(values)),
                "min_cosine_displacement": float(np.min(values)),
                "max_cosine_displacement": float(np.max(values)),
            }
        )
    return summary


def write_displacement_summary(
    config: dict[str, Any],
    paths: PipelinePaths,
) -> tuple[Path, int] | None:
    if not enabled(config):
        return None
    destination = output_path(config, paths, "displacement_summary")
    rows = _displacement_summary(read_tsv(paths.displacement_path))
    write_tsv(
        rows,
        destination,
        [
            "scope",
            "n_terms",
            "mean_cosine_displacement",
            "median_cosine_displacement",
            "min_cosine_displacement",
            "max_cosine_displacement",
        ],
    )
    return destination, len(rows)


def expected_paths(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Path]:
    return {
        key: output_path(config, paths, key)
        for key in EXAMPLE_FILENAMES
    }


def relative_output_dir(config: dict[str, Any], paths: PipelinePaths) -> str:
    return Path(os.path.relpath(output_dir(config, paths), paths.run_dir)).as_posix()
