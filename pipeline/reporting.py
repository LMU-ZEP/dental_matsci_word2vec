from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .io_utils import json_safe


def write_stage_report(
    *,
    report_path: str | Path,
    title: str,
    purpose: str,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metrics: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    base = report_path.parent.parent.resolve()

    def display_path(value: str | Path) -> str:
        path = Path(value)
        try:
            return Path(os.path.relpath(path.resolve(), base)).as_posix()
        except OSError:
            return path.as_posix()

    lines = [f"# {title}", "", purpose, "", "## Inputs", ""]
    lines.extend(f"- `{display_path(p)}`" for p in inputs)
    lines.extend(["", "## Outputs", ""])
    lines.extend(f"- `{display_path(p)}`" for p in outputs)
    lines.extend(["", "## Metrics", "", "```json"])
    lines.append(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2))
    lines.append("```")

    if parameters:
        lines.extend(["", "## Parameters", "", "```json"])
        lines.append(json.dumps(json_safe(parameters), ensure_ascii=False, indent=2))
        lines.append("```")

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
