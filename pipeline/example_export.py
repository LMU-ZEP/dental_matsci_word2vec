from __future__ import annotations

from typing import Any

from .config import PipelinePaths
from .example_outputs import enabled, expected_paths, relative_output_dir
from .reporting import write_stage_report


def run(config: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    settings = config.get("example_export", {})
    if not enabled(config):
        metrics = {"enabled": False}
        write_stage_report(
            report_path=paths.reports_dir / "09_example_export.md",
            title="Stage 9 — Example artifact verification",
            purpose="Verifies the reviewer-facing example artifacts written by prior stages.",
            inputs=[],
            outputs=[],
            metrics=metrics,
            parameters=settings,
            notes=["Example output was disabled in the configuration."],
        )
        return metrics

    outputs = expected_paths(config, paths)
    missing = [path.as_posix() for path in outputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Example artifacts are incomplete. Missing: " + ", ".join(missing)
        )

    metrics = {
        "enabled": True,
        "n_expected_files": len(outputs),
        "n_existing_files": sum(path.exists() for path in outputs.values()),
        "files": {key: path.name for key, path in outputs.items()},
    }
    write_stage_report(
        report_path=paths.reports_dir / "09_example_export.md",
        title="Stage 9 — Example artifact verification",
        purpose=(
            "Verifies that every preceding stage wrote its reviewer-facing "
            "example artifact with the requested filename."
        ),
        inputs=list(outputs.values()),
        outputs=list(outputs.values()),
        metrics=metrics,
        parameters=settings | {"output_dir": relative_output_dir(config, paths)},
        notes=[
            "The files are produced incrementally: manifest, preprocessing, phrases, "
            "training, alignment, and evidence each write their own example artifact."
        ],
    )
    return metrics
