#!/usr/bin/env python3
"""Regression checks against preserved real outputs. Does NOT retrain Word2Vec."""
from __future__ import annotations
import json, math, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/"validation/fixtures/real_outputs"
EXPECTED=json.loads((FIX/"2D_authoritative_summary.json").read_text())

def close(a,b,tol=1e-10):
    return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)

def assert_close(a,b,label,tol=1e-10):
    if not close(a,b,tol): raise AssertionError(f"{label}: got {a}, expected {b}")

def main():
    # 1. Recompute 30k/50k sensitivity from preserved real displacement rows.
    d30=pd.read_csv(FIX/"cosine_displacement_30k.csv")
    d50=pd.read_csv(FIX/"cosine_displacement_50k.csv")
    a=d30[d30.status.eq("compared")].drop_duplicates("term")[["term","cosine_displacement"]]
    b=d50[d50.status.eq("compared")].drop_duplicates("term")[["term","cosine_displacement"]]
    m=a.merge(b,on="term",suffixes=("_30k","_50k"),validate="one_to_one")
    m["abs_diff"]=(m.cosine_displacement_30k-m.cosine_displacement_50k).abs()
    exp=EXPECTED["alignment_vocabulary_sensitivity"]
    assert len(m)==exp["n_unique_compared_terms"]==53
    assert_close(pearsonr(m.cosine_displacement_30k,m.cosine_displacement_50k).statistic, exp["pearson_r"], "Pearson")
    assert_close(spearmanr(m.cosine_displacement_30k,m.cosine_displacement_50k).statistic, exp["spearman_rho"], "Spearman")
    assert_close(m.abs_diff.mean(), exp["mean_absolute_difference"], "mean abs diff")
    assert_close(m.abs_diff.median(), exp["median_absolute_difference"], "median abs diff")
    assert_close(m.abs_diff.max(), exp["max_absolute_difference"], "max abs diff")

    # 2. Exercise the repository sensitivity CLI on the same real rows.
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/"sensitivity.csv"
        p=subprocess.run([
            sys.executable, str(ROOT/"scripts/analysis/calculate_procrustes_sensitivity.py"),
            "--displacement-30k", str(FIX/"cosine_displacement_30k.csv"),
            "--displacement-50k", str(FIX/"cosine_displacement_50k.csv"),
            "--output-csv", str(out),
        ],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode: raise AssertionError(p.stderr)
        outdf=pd.read_csv(out)
        assert len(outdf)==53 and outdf.term.nunique()==53

        # 3. Exercise canonical summary CLI against the real 30k displacement file.
        sumdir=Path(td)/"summary"
        p=subprocess.run([
            sys.executable, str(ROOT/"scripts/analysis/calculate_procrustes_summary.py"),
            "--displacement-csv", str(FIX/"cosine_displacement_30k.csv"),
            "--target-terms", str(ROOT/"configs/target_terms.tsv"),
            "--out-dir", str(sumdir),
            "--reliable-min-count", "1000",
        ],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode: raise AssertionError(p.stderr)
        gs=pd.read_csv(sumdir/"cosine_displacement_global_summary.csv")
        u=gs.loc[gs.summary_variant.eq("unique_terms")].iloc[0]
        r=gs.loc[gs.summary_variant.eq("unique_terms_min_count_ge_1000")].iloc[0]
        eall=EXPECTED["reliability_filter"]["all_unique_terms"]
        erel=EXPECTED["reliability_filter"]["min_count_ge_1000"]
        assert int(u.n_terms)==eall["n_terms"]==53
        assert int(r.n_terms)==erel["n_terms"]==43
        assert_close(u.mean_displacement,eall["mean_displacement"],"global mean")
        assert_close(u.median_displacement,eall["median_displacement"],"global median")
        assert_close(u.std_displacement,eall["sd_displacement"],"global SD")
        assert_close(r.mean_displacement,erel["mean_displacement"],"reliable mean")
        assert_close(r.median_displacement,erel["median_displacement"],"reliable median")
        assert_close(r.std_displacement,erel["sd_displacement"],"reliable SD")

    # 4. Preserved held-out diagnostic values.
    diag=pd.read_csv(FIX/"alignment_diagnostic_summary.csv").iloc[0]
    ed=EXPECTED["alignment_diagnostic"]
    mapping={
        "n_alignment_train_terms":"diagnostic_fit_terms",
        "n_alignment_eval_terms":"heldout_terms",
        "median_cosine_before":"median_cosine_distance_before",
        "median_cosine_after":"median_cosine_distance_after",
        "median_improvement":"median_per_term_improvement",
        "top1_accuracy":"top1_accuracy","top5_accuracy":"top5_accuracy","top10_accuracy":"top10_accuracy",
        "orthogonality_error":"orthogonality_error",
    }
    for col,key in mapping.items(): assert_close(diag[col],ed[key],f"diagnostic {col}")

    # 5. Preserved final 31-anchor consistency summary.
    cons=json.loads((FIX/"consistency_summary_31anchors.json").read_text())
    assert len(cons)==2
    for got,ex in zip(cons,EXPECTED["consistency_90pct"]):
        assert got["label"]==ex["label"]
        assert got["n_anchors_total"]==31
        assert got["n_anchors_present_in_both"]==ex["n_anchors_present_in_both"]
        for k in ["10","20"]: assert_close(got["mean_jaccard"][k],ex["mean_jaccard"][k],f"{got['label']} Jaccard@{k}")
        assert_close(got["pairwise_similarity_consistency"]["pearson_pairwise_similarity"], ex["pairwise_similarity_consistency"]["pearson_pairwise_similarity"], f"{got['label']} Pearson")

    print("Regression PASS: real saved outputs reproduce canonical 2D statistics without retraining models.")
    print("  targets: 53 unique; reliable: 43")
    print("  30k vs 50k: Pearson 0.9981916; Spearman 0.9928237")
    print("  held-out: 24,000 fit / 6,000 held-out; top-1 0.883")
    print("  consistency: pre 27/31; post 31/31")

if __name__=="__main__": main()
