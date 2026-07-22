#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from pipeline import alignment, evidence, example_export, extraction, manifest
from pipeline import phrases, preprocessing, visualization, word2vec_training
from pipeline.config import DEFAULT_STAGES, PipelinePaths, load_config, save_resolved_config
from pipeline.example_outputs import copy_config


STAGE_FUNCTIONS: dict[str, Callable] = {
    "manifest": manifest.run,
    "extraction": extraction.run,
    "preprocessing": preprocessing.run,
    "phrases": phrases.run,
    "training": word2vec_training.run,
    "alignment": alignment.run,
    "evidence": evidence.run,
    "visualization": visualization.run,
    "example_export": example_export.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the modular PDF/XML → Word2Vec → semantic-shift pipeline."
    )
    parser.add_argument("--config", help="Path to a JSON pipeline config.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGE_FUNCTIONS),
        help="Run only selected stages. Dependencies must already exist.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print stage names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_stages:
        print("\n".join(STAGE_FUNCTIONS))
        return

    if not args.config:
        raise SystemExit("--config is required unless --list-stages is used.")

    config, config_path = load_config(args.config)
    paths = PipelinePaths.from_config(config, config_path)
    paths.ensure_directories()
    save_resolved_config(config, paths.resolved_config_path)
    copy_config(config_path, config, paths)

    stages = args.stages or config.get("stages", DEFAULT_STAGES)
    unknown = [stage for stage in stages if stage not in STAGE_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")

    run_summary = {}
    for stage in stages:
        print(f"\n=== {stage} ===")
        run_summary[stage] = STAGE_FUNCTIONS[stage](config, paths)

    summary_path = paths.run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nPipeline complete. Summary: {summary_path}")
    print(f"Stage reports: {paths.reports_dir}")


if __name__ == "__main__":
    main()
