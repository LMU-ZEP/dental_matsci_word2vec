from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STAGES = [
    "manifest",
    "extraction",
    "preprocessing",
    "phrases",
    "training",
    "alignment",
    "evidence",
    "visualization",
    "example_export",
]


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    run_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    manifest_path: Path
    raw_documents_path: Path
    preprocessed_path: Path
    phrased_path: Path
    phraser_path: Path
    phrase_counts_path: Path
    models_dir: Path
    vocabulary_counts_path: Path
    alignment_dir: Path
    alignment_npz_path: Path
    alignment_terms_path: Path
    displacement_path: Path
    evidence_dir: Path
    neighbors_path: Path
    jaccard_path: Path
    emergence_path: Path
    visualization_dir: Path
    resolved_config_path: Path

    @classmethod
    def from_config(cls, config: dict[str, Any], config_path: Path) -> "PipelinePaths":
        project_root = config_path.parent.resolve()
        output_dir = Path(config.get("output_dir", "outputs/run"))
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()

        artifacts = output_dir / "artifacts"
        alignment = artifacts / "06_alignment"
        evidence = artifacts / "07_evidence"

        return cls(
            project_root=project_root,
            run_dir=output_dir,
            artifacts_dir=artifacts,
            reports_dir=output_dir / "reports",
            manifest_path=artifacts / "01_manifest" / "source_manifest.jsonl",
            raw_documents_path=artifacts / "02_extraction" / "raw_documents.jsonl",
            preprocessed_path=artifacts / "03_preprocessing" / "preprocessed_tokens.jsonl",
            phrased_path=artifacts / "04_phrases" / "phrased_tokens.jsonl",
            phraser_path=artifacts / "04_phrases" / "shared_phraser.phraser",
            phrase_counts_path=artifacts / "04_phrases" / "phrase_counts.tsv",
            models_dir=artifacts / "05_models",
            vocabulary_counts_path=artifacts / "05_models" / "vocabulary_counts.tsv",
            alignment_dir=alignment,
            alignment_npz_path=alignment / "procrustes_alignment.npz",
            alignment_terms_path=alignment / "alignment_terms.txt",
            displacement_path=alignment / "cosine_displacement.tsv",
            evidence_dir=evidence,
            neighbors_path=evidence / "neighbors.tsv",
            jaccard_path=evidence / "jaccard_overlap.tsv",
            emergence_path=evidence / "vocabulary_emergence.tsv",
            visualization_dir=artifacts / "08_visualization",
            resolved_config_path=output_dir / "resolved_config.json",
        )

    def model_path(self, period: str) -> Path:
        return self.models_dir / f"word2vec_{period}.model"

    def training_config_path(self, period: str) -> Path:
        return self.models_dir / f"word2vec_{period}.config.json"

    def ensure_directories(self) -> None:
        for path in [
            self.run_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.manifest_path.parent,
            self.raw_documents_path.parent,
            self.preprocessed_path.parent,
            self.phrased_path.parent,
            self.models_dir,
            self.alignment_dir,
            self.evidence_dir,
            self.visualization_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _resolve_path(value: str | Path, base: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.as_posix()


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Resolve path-bearing config fields while preserving inline example data."""
    base = config_path.parent.resolve()
    resolved = json.loads(json.dumps(config))

    if "output_dir" in resolved:
        resolved["output_dir"] = _resolve_path(resolved["output_dir"], base)

    for period, sources in resolved.get("periods", {}).items():
        for source in sources:
            if "path" in source:
                source["path"] = _resolve_path(source["path"], base)
            if "paths" in source:
                source["paths"] = [_resolve_path(p, base) for p in source["paths"]]

    analysis = resolved.get("analysis", {})
    if analysis.get("target_terms_path"):
        analysis["target_terms_path"] = _resolve_path(analysis["target_terms_path"], base)

    export = resolved.get("example_export", {})
    if export.get("output_dir"):
        export["output_dir"] = _resolve_path(export["output_dir"], base)

    return resolved


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = resolve_config_paths(config, config_path)
    validate_config(config)
    return config, config_path


def validate_config(config: dict[str, Any]) -> None:
    periods = config.get("periods", {})
    if len(periods) < 2:
        raise ValueError("Config must define at least two periods under 'periods'.")

    training = config.get("word2vec", {})
    if int(training.get("vector_size", 200)) <= 0:
        raise ValueError("word2vec.vector_size must be positive.")

    if not (
        config.get("analysis", {}).get("target_terms")
        or config.get("analysis", {}).get("target_terms_path")
    ):
        raise ValueError(
            "Provide analysis.target_terms or analysis.target_terms_path."
        )


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
