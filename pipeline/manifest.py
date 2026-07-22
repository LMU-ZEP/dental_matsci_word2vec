from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from .config import PipelinePaths
from .example_outputs import copy_manifest
from .io_utils import file_size, sha256_file, stable_id, write_jsonl
from .reporting import write_stage_report


def _glob_source(source: dict[str, Any]) -> list[Path]:
    raw_paths: list[str] = []
    if source.get("path"):
        raw_paths.append(source["path"])
    raw_paths.extend(source.get("paths", []))

    source_type = source["type"].lower()
    default_pattern = "*.pdf" if source_type == "pdf" else "*.xml"
    pattern = source.get("glob", default_pattern)
    recursive = bool(source.get("recursive", True))

    found: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(path.rglob(pattern) if recursive else path.glob(pattern))
        else:
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in found})


def _selected_paths(
    paths: list[Path],
    fraction: float,
    seed: int,
) -> set[Path]:
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    if fraction == 1 or not paths:
        return set(paths)
    n = max(1, int(len(paths) * fraction))
    return set(random.Random(seed).sample(paths, n))


def _file_records(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base_seed = int(config.get("seed", 42))
    hash_files = bool(config.get("manifest", {}).get("hash_files", True))

    for period_index, (period, sources) in enumerate(config["periods"].items()):
        for source_index, source in enumerate(sources):
            source_type = source["type"].lower()
            if source_type == "inline":
                for item_index, item in enumerate(source.get("documents", [])):
                    source_path = item.get(
                        "source_path", f"inline://{period}/{source_index}/{item_index}"
                    )
                    yield {
                        "document_id": stable_id(period, source_type, source_path),
                        "period": period,
                        "source_type": item.get("source_type", "inline"),
                        "ingest_type": "inline",
                        "source_path": source_path,
                        "publication_year": item.get("publication_year"),
                        "selected": True,
                        "status": "selected",
                        "file_size_bytes": None,
                        "sha256": None,
                        "inline_text": item["text"],
                    }
                continue

            paths = _glob_source(source)
            fraction = float(source.get("sample_fraction", 1.0))
            selected = _selected_paths(
                paths,
                fraction,
                base_seed + period_index * 10_000 + source_index,
            )
            for path in paths:
                is_selected = path in selected
                yield {
                    "document_id": stable_id(period, source_type, path.as_posix()),
                    "period": period,
                    "source_type": source_type,
                    "ingest_type": source_type,
                    "source_path": path.as_posix(),
                    "publication_year": source.get("publication_year"),
                    "selected": is_selected,
                    "status": "selected" if is_selected else "not_selected",
                    "file_size_bytes": file_size(path),
                    "sha256": sha256_file(path) if hash_files else None,
                }


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    records = list(_file_records(config))
    write_jsonl(records, paths.manifest_path)
    example_path = copy_manifest(config, paths)
    metrics = {
        "n_manifest_records": len(records),
        "n_selected": sum(bool(row["selected"]) for row in records),
        "period_counts": {
            period: sum(row["period"] == period and row["selected"] for row in records)
            for period in config["periods"]
        },
        "source_type_counts": {
            source_type: sum(
                row["source_type"] == source_type and row["selected"]
                for row in records
            )
            for source_type in sorted({row["source_type"] for row in records})
        },
    }
    write_stage_report(
        report_path=paths.reports_dir / "01_manifest.md",
        title="Stage 1 — Source manifest",
        purpose=(
            "Enumerates every source deterministically and records the exact subset "
            "selected for this run."
        ),
        inputs=[],
        outputs=[paths.manifest_path] + ([example_path] if example_path else []),
        metrics=metrics,
        parameters=config.get("manifest", {}),
    )
    return metrics
