#!/usr/bin/env python3
from __future__ import annotations
import compileall, json, os, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLI_SCRIPTS = [
    "scripts/corpus/get_valid_pdfs.py",
    "scripts/corpus/get_xml_corpus.py",
    "scripts/corpus/word2vec_pipeline_step2_xml_pdf.py",
    "scripts/modeling/build_shared_phraser.py",
    "scripts/modeling/train_word2vec_from_tokens.py",
    "scripts/analysis/procrustes_displacement.py",
    "scripts/analysis/procrustes_diagnostic.py",
    "scripts/analysis/calculate_procrustes_summary.py",
    "scripts/analysis/calculate_procrustes_sensitivity.py",
    "scripts/analysis/semantic_shift_evidence.py",
    "scripts/robustness/sample90_and_train_word2vec.py",
    "scripts/robustness/compare_word2vec_consistency.py",
]

EXPECTED_W2V = {
    "vector_size": 200, "window": 8, "min_count": 20, "epochs": 15,
    "sample": 1e-4, "negative": 15, "alpha": 0.025,
    "min_alpha": 0.0005, "sg": 1, "skip_gram": True,
    "seed": 42, "workers": 1, "deterministic": True,
}

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    results=[]
    ok = compileall.compile_dir(ROOT / "scripts", quiet=1) and compileall.compile_dir(ROOT / "audits", quiet=1)
    check(ok, "compileall failed")
    results.append("compileall: PASS")

    # --help must be side-effect free. Heavy runtime resources should not be needed.
    for rel in CLI_SCRIPTS:
        p = subprocess.run([sys.executable, str(ROOT/rel), "--help"], cwd=ROOT,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        check(p.returncode == 0, f"{rel} --help failed: {p.stderr[:500]}")
    results.append(f"CLI --help: PASS ({len(CLI_SCRIPTS)} scripts)")

    # No executable repository script should embed a user-specific /home/... production path.
    offenders=[]
    for base in [ROOT/"scripts", ROOT/"audits"]:
        for p in base.rglob("*.py"):
            txt=p.read_text(encoding="utf-8", errors="replace")
            if "/home/" in txt:
                offenders.append(str(p.relative_to(ROOT)))
    check(not offenders, "hard-coded /home paths: " + ", ".join(offenders))
    results.append("hard-coded /home paths: PASS")

    targets=pd.read_csv(ROOT/"configs/target_terms.tsv", sep="\t")
    check(len(targets)==56, f"target memberships={len(targets)}, expected 56")
    check(targets.term.nunique()==53, f"unique targets={targets.term.nunique()}, expected 53")
    results.append("canonical targets: PASS (56 memberships / 53 unique)")

    anchors=[x.strip() for x in (ROOT/"configs/anchor_terms_consistency.txt").read_text().splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    check(len(anchors)==31 and len(set(anchors))==31, f"anchors={len(anchors)}/{len(set(anchors))}, expected 31 unique")
    results.append("consistency anchors: PASS (31 unique)")

    for name in ["matsci_pre2018_sharedphrases.config.json", "matsci_post2018_sharedphrases.config.json"]:
        cfg=json.loads((ROOT/"run_metadata"/name).read_text())
        got=cfg["word2vec"]
        for k,v in EXPECTED_W2V.items():
            check(got[k]==v, f"{name}: {k}={got[k]!r}, expected {v!r}")
    results.append("actual Word2Vec run configs: PASS and identical to manuscript parameters")

    # Environment resources are reported separately; absent resources are not repository-code failures.
    env=[]
    for mod in ["numpy","pandas","scipy","sklearn","gensim","spacy","nltk","pypdf","fitz"]:
        try:
            __import__(mod); env.append(f"{mod}=OK")
        except Exception as e:
            env.append(f"{mod}=MISSING({type(e).__name__})")
    try:
        import ijson; env.append("ijson=OK")
    except Exception: env.append("ijson=MISSING")
    try:
        import spacy; spacy.load("en_core_web_sm", disable=["parser","ner"]); env.append("en_core_web_sm=OK")
    except Exception: env.append("en_core_web_sm=MISSING")
    try:
        from nltk.corpus import stopwords; stopwords.words("english"); env.append("nltk_stopwords=OK")
    except Exception: env.append("nltk_stopwords=MISSING")

    print("\n".join(results))
    print("environment: " + "; ".join(env))
    print("SMOKE TEST PASS")

if __name__ == "__main__": main()
