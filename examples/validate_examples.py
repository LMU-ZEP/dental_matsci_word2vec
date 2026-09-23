#!/usr/bin/env python3
"""Validate fixtures, actual output snapshots, and generated intermediates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
SNAP = EX / "manuscript_output_snapshots"
GEN = EX / "inspectable_pipeline"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def validate_snapshots() -> None:
    unique = pd.read_csv(SNAP / "cosine_displacement_unique_terms.csv")
    reliable = pd.read_csv(SNAP / "cosine_displacement_reliable_unique_terms.csv")
    assert len(unique) == 53 and unique["term"].nunique() == 53
    assert len(reliable) == 43 and reliable["term"].nunique() == 43

    global_summary = pd.read_csv(SNAP / "cosine_displacement_global_summary.csv")
    row = global_summary.loc[global_summary["summary_variant"] == "unique_terms"].iloc[0]
    assert int(row["n_terms"]) == 53
    assert close(row["mean_displacement"], 0.16606631009193604)
    assert close(row["median_displacement"], 0.134409191133656)
    row_rel = global_summary.loc[global_summary["summary_variant"] == "unique_terms_min_count_ge_1000"].iloc[0]
    assert int(row_rel["n_terms"]) == 43
    assert close(row_rel["mean_displacement"], 0.13582866096729232)
    assert close(row_rel["std_displacement"], 0.04415646394498862)

    diag = pd.read_csv(SNAP / "alignment_diagnostic_summary.csv").iloc[0]
    assert int(diag["n_alignment_train_terms"]) == 24000
    assert int(diag["n_alignment_eval_terms"]) == 6000
    assert close(diag["top1_accuracy"], 0.883)
    assert close(diag["top5_accuracy"], 0.9683333333333334)
    assert close(diag["top10_accuracy"], 0.9803333333333333)

    sens = pd.read_csv(SNAP / "procrustes_30k_50k_sensitivity.csv")
    assert len(sens) == 53 and sens["term"].nunique() == 53
    pear = pearsonr(sens["cosine_displacement_30k"], sens["cosine_displacement_50k"]).statistic
    spear = spearmanr(sens["cosine_displacement_30k"], sens["cosine_displacement_50k"]).statistic
    assert close(pear, 0.99819163296, 1e-8)
    assert close(spear, 0.99282373811, 1e-8)

    nn = pd.read_csv(SNAP / "nearest_neighbor_table_common_vocab.csv")
    jac = pd.read_csv(SNAP / "jaccard_overlap_common_vocab.csv")
    assert len(nn) == 53 and nn["term"].nunique() == 53
    assert len(jac) == 159 and jac["term"].nunique() == 53
    assert set(jac["k"].astype(int)) == {5, 10, 20}

    consistency = json.loads((SNAP / "consistency_summary_31anchors.json").read_text(encoding="utf-8"))
    assert [x["n_anchors_total"] for x in consistency] == [31, 31]
    assert [x["n_anchors_present_in_both"] for x in consistency] == [27, 31]


def validate_fixtures() -> None:
    xmls = sorted((EX / "fixtures").glob("xml_*/*.xml"))
    pdf_txts = sorted((EX / "fixtures/pdf").glob("*.txt"))
    pdfs = sorted((EX / "fixtures/pdf").glob("*.pdf"))
    assert len(xmls) == 8
    assert len(pdf_txts) == 2 and len(pdfs) == 2
    for p in xmls + pdf_txts:
        txt = p.read_text(encoding="utf-8")
        assert "Synthetic" in txt or "synthetic" in txt


def validate_generated(require: bool) -> bool:
    manifest_path = GEN / "MANIFEST.json"
    if not manifest_path.exists():
        if require:
            raise AssertionError(
                "Generated inspectable pipeline is missing. Run: python examples/build_inspectable_examples.py"
            )
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["synthetic_sources_only"] is True
    assert manifest["production_preprocessing_exercised"] is True
    required = [
        "01_sources_and_extraction/source_manifest.jsonl",
        "02_preprocessing/pre_preprocessed.json",
        "02_preprocessing/post_preprocessed.json",
        "03_shared_phrases/pre_shared_phrased.json",
        "03_shared_phrases/post_shared_phrased.json",
        "04_word2vec/pre_demo.vocab.csv",
        "04_word2vec/post_demo.vocab.csv",
        "04_word2vec/trained_vectors_sample.csv",
        "05_procrustes/cosine_displacement.csv",
        "06_semantic_evidence/nearest_neighbor_table_common_vocab.csv",
        "06_semantic_evidence/jaccard_overlap_common_vocab.csv",
        "reports/01_sources_and_extraction.md",
        "reports/02_preprocessing.md",
        "reports/03_shared_phrases.md",
        "reports/04_word2vec_training.md",
        "reports/05_procrustes_alignment.md",
        "reports/06_semantic_evidence.md",
        "reports/07_relation_to_manuscript.md",
    ]
    for rel in required:
        assert (GEN / rel).is_file(), rel

    by_path = {x["path"]: x for x in manifest["files"]}
    for rel, rec in by_path.items():
        p = GEN / rel
        assert p.is_file(), rel
        assert p.stat().st_size == int(rec["bytes"]), rel
        assert sha256(p) == rec["sha256"], rel
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-generated", action="store_true")
    args = ap.parse_args()
    validate_fixtures()
    validate_snapshots()
    generated = validate_generated(args.require_generated)
    print("Inspectable-example validation PASS")
    print("  synthetic fixtures: PASS")
    print("  actual manuscript-output snapshots: PASS")
    print("  generated inspectable pipeline:", "PASS" if generated else "NOT YET GENERATED")


if __name__ == "__main__":
    main()
