#!/usr/bin/env python3
"""Miniature end-to-end validation on synthetic, redistributable data.

Always validates XML extraction + normalization + shared phrases + Word2Vec +
Procrustes + displacement/Jaccard. If the full spaCy/NLTK resources are installed,
it additionally exercises the production preprocessing function from raw text.
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMPAT=ROOT/"validation/compat"
PYENV=os.environ.copy()
try:
    import ijson as _real_ijson  # noqa: F401
except ImportError:
    # Test-only fallback for tiny fixtures; full production runs still require ijson.
    PYENV["PYTHONPATH"] = str(COMPAT) + os.pathsep + PYENV.get("PYTHONPATH","")
PYENV["PYTHONHASHSEED"]="42"

def run(cmd,cwd=ROOT):
    p=subprocess.run(cmd,cwd=cwd,env=PYENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("COMMAND FAILED\n"+" ".join(map(str,cmd))+"\nSTDOUT:\n"+p.stdout+"\nSTDERR:\n"+p.stderr)
    return p

def write_xml(path, year, title, body):
    path.write_text(f'''<root><article><head><title>{title}</title><cover-date-year>{year}</cover-date-year><abstract><simple-para>{body}</simple-para></abstract></head><body><sections><section><section-title>Results</section-title><para>{body}</para></section></sections></body></article></root>''',encoding="utf-8")

def synthetic_tokens(period):
    bg=["dental","material","study","ceramic","resin","mechanical","optical","surface","strength","testing","clinical","restoration","specimen","analysis","property","structure","method","result","sample","performance","fracture","color","phase","matrix","filler","bond","wear","hardness","modulus","roughness"]
    docs=[]
    for i in range(90):
        toks=bg.copy()
        toks += ["fracture","toughness","crack","bridging","shade","matching","color","matching","monolithic","zirconia","yttria","translucency"]
        if period=="pre": toks += ["framework","opaque","tetragonal"]*2
        else: toks += ["translucent","monolithic","yttria","digital"]*2
        # deterministic rotations add context diversity without changing vocabulary coverage
        r=i%len(toks); toks=toks[r:]+toks[:r]
        docs.append(toks)
    return docs

def main():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); (td/"xml_pre").mkdir(); (td/"xml_post").mkdir()

        # Use several small XML records with a deliberately rich shared vocabulary.
        # This keeps the production-preprocessing branch large enough for the
        # downstream Word2Vec/Procrustes smoke analysis while still being tiny.
        shared = (
            "Dental material zirconia fracture toughness crack bridging shade matching "
            "color matching resin composite optical strength surface ceramic testing "
            "clinical restoration structure property filler matrix wear hardness modulus."
        )
        for i, year in enumerate(range(2014, 2018), start=1):
            write_xml(
                td/"xml_pre"/f"pre_{i}.xml",
                year,
                f"Dental materials study {i}",
                shared + " Framework opaque tetragonal conventional processing.",
            )
        for i, year in enumerate(range(2019, 2023), start=1):
            write_xml(
                td/"xml_post"/f"post_{i}.xml",
                year,
                f"Dental materials study {i}",
                shared + " Monolithic translucent digital cubic yttria processing.",
            )

        # Actual XML extraction CLI.
        run([sys.executable,str(ROOT/"scripts/corpus/get_xml_corpus.py"),"--xml-inputs",str(td/"xml_pre"),"--output-json",str(td/"pre_raw.json"),"--min-year","1992"])
        run([sys.executable,str(ROOT/"scripts/corpus/get_xml_corpus.py"),"--xml-inputs",str(td/"xml_post"),"--output-json",str(td/"post_raw.json"),"--min-year","2018"])
        assert json.loads((td/"pre_raw.json").read_text()) and json.loads((td/"post_raw.json").read_text())

        # Actual text normalization module.
        sys.path.insert(0,str(ROOT/"scripts/corpus"))
        import text_normalization as tn
        assert tn.cleanup_extracted_text("4Y–PSZ and 3Y-\nTZP") == "4Y-PSZ and 3Y-TZP"

        # Full production preprocessing if resources exist; otherwise report a controlled skip.
        # Resource discovery is separated from execution so a real preprocessing error
        # is not silently converted into a SKIPPED result.
        preprocessing="SKIPPED (spaCy model/NLTK stopwords unavailable in this environment)"
        production_resources_available=False
        try:
            import spacy
            from nltk.corpus import stopwords
            spacy.load("en_core_web_sm",disable=["parser","ner"])
            stopwords.words("english")
            production_resources_available=True
        except (ImportError, OSError, LookupError):
            production_resources_available=False

        if production_resources_available:
            # Production preprocessing script intentionally stops after preprocessing.
            # Any command failure here is a validation failure and must propagate.
            for label in ["pre","post"]:
                run([sys.executable,str(ROOT/"scripts/corpus/word2vec_pipeline_step2_xml_pdf.py"),
                     "--corpus-name","mini","--period",label,"--reuse-raw-corpus",
                     "--raw-corpus-path",str(td/f"{label}_raw.json"),
                     "--preprocessed-corpus-path",str(td/f"{label}_preprocessed.json"),
                     "--output-dir",str(td/f"prep_{label}")])
            preprocessing="PASS (production spaCy/NLTK preprocessing exercised)"
        else:
            # Continue with explicit synthetic token corpora to validate every downstream stage.
            (td/"pre_preprocessed.json").write_text(json.dumps(synthetic_tokens("pre")),encoding="utf-8")
            (td/"post_preprocessed.json").write_text(json.dumps(synthetic_tokens("post")),encoding="utf-8")

        # Shared phrase model, applied unchanged to both periods.
        run([sys.executable,str(ROOT/"scripts/modeling/build_shared_phraser.py"),
             "--input-corpora",str(td/"pre_preprocessed.json"),str(td/"post_preprocessed.json"),
             "--output-corpora",str(td/"pre_phrased.json"),str(td/"post_phrased.json"),
             "--phraser-path",str(td/"shared.phraser"),"--phrase-min-count","2","--phrase-threshold","0.1",
             "--summary-json",str(td/"phrase_summary.json")])
        preprocessed_pre=json.loads((td/"pre_preprocessed.json").read_text())
        preprocessed_post=json.loads((td/"post_preprocessed.json").read_text())
        prephr=json.loads((td/"pre_phrased.json").read_text())
        postphr=json.loads((td/"post_phrased.json").read_text())
        # Phrase detection must preserve the number of corpus units.  Do not
        # assume 90 here: the synthetic fallback contains 90 units, whereas
        # successful production preprocessing operates on the tiny XML fixture
        # created above and therefore yields a different (but valid) count.
        assert len(preprocessed_pre) > 0 and len(preprocessed_post) > 0
        assert len(prephr) == len(preprocessed_pre)
        assert len(postphr) == len(preprocessed_post)

        # Deterministic Word2Vec on both phrased corpora.
        for label in ["pre","post"]:
            run([sys.executable,str(ROOT/"scripts/modeling/train_word2vec_from_tokens.py"),
                 "--corpus",str(td/f"{label}_phrased.json"),"--model-path",str(td/f"{label}.model"),
                 "--vocab-path",str(td/f"{label}.vocab.csv"),"--config-path",str(td/f"{label}.config.json"),
                 "--seed","42","--deterministic","--vector-size","30","--window","5","--min-count","1",
                 "--epochs","20","--sample","0.001","--negative","5","--skip-gram"])

        # Select three target terms from the ACTUAL shared vocabulary.  Prefer
        # domain terms, but do not assume a particular phrase segmentation: gensim
        # phrase scoring can vary slightly across versions.  The validation goal here
        # is execution/data-flow correctness, not the scientific identity of mini targets.
        from gensim.models import Word2Vec
        import re
        pre=Word2Vec.load(str(td/"pre.model")); post=Word2Vec.load(str(td/"post.model"))
        common=set(pre.wv.index_to_key) & set(post.wv.index_to_key)
        clean_common=[
            w for w in common
            if len(w)>=3 and re.search(r"[A-Za-z]", w) and not re.fullmatch(r"\d+", w)
        ]
        clean_common.sort(key=lambda w: (-min(int(pre.wv.get_vecattr(w,"count")), int(post.wv.get_vecattr(w,"count"))), w))

        preferred=[
            "fracture_toughness", "crack_bridging", "shade_matching",
            "color_matching", "zirconia", "strength", "optical",
            "resin_composite", "fracture", "shade", "color",
        ]
        selected=[]
        for term in preferred + clean_common:
            if term in common and term not in selected:
                selected.append(term)
            if len(selected)==3:
                break

        # Keep at least three clean shared terms outside the targets for alignment.
        background=[w for w in clean_common if w not in selected]
        if len(selected)<3 or len(background)<3:
            raise AssertionError(
                "Mini corpus did not retain enough clean shared vocabulary after phrase detection: "
                f"shared={len(common)}, clean_shared={len(clean_common)}, "
                f"selected={selected}, background={background[:20]}"
            )

        groups=["zirconia","short_fiber_composite","structural_color_composite"]
        available=list(zip(selected, groups))
        with (td/"targets.tsv").open("w") as f:
            f.write("term\tmaterial_system\n")
            for t,g in available:
                f.write(f"{t}\t{g}\n")
        # visualization file must contain all canonical targets for the same groups.
        (td/"visualization.tsv").write_text((td/"targets.tsv").read_text())

        run([sys.executable,str(ROOT/"scripts/analysis/procrustes_displacement.py"),
             "--pre-model",str(td/"pre.model"),"--post-model",str(td/"post.model"),
             "--target-terms",str(td/"targets.tsv"),"--out-dir",str(td/"proc"),"--min-count","1","--top-n","15"])
        assert (td/"proc/cosine_displacement.csv").exists() and (td/"proc/procrustes_alignment.npz").exists()
        proc_meta=json.loads((td/"proc/procrustes_metadata.json").read_text())
        assert proc_meta["n_alignment_terms"] >= 3
        assert proc_meta["n_compared_terms"] == 3

        # Actual nearest-neighbor/Jaccard implementation, without running the optional
        # dimensionality-reduction plotting branch (keeps the validation fast).
        import importlib.util
        spec=importlib.util.spec_from_file_location("semantic_shift_evidence", ROOT/"scripts/analysis/semantic_shift_evidence.py")
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        terms_df=mod.read_terms_table(td/"targets.tsv")
        pre_wv=mod.load_wv(td/"pre.model"); post_wv=mod.load_wv(td/"post.model")
        evidence_dir=td/"evidence"; evidence_dir.mkdir()
        mod.compute_jaccard_and_neighbors(pre_wv,post_wv,terms_df,evidence_dir,k_values=(5,10),nn_topn=10,mode="common_vocab")
        assert (evidence_dir/"jaccard_overlap_common_vocab.csv").exists()

        print("Mini E2E PASS")
        print("  XML extraction: PASS")
        print("  text normalization: PASS")
        print("  production preprocessing:",preprocessing)
        print("  shared phrase detection: PASS")
        print("  deterministic Word2Vec: PASS")
        print("  shared vocabulary:",len(common),"terms; mini targets:",", ".join(selected))
        print("  Procrustes + displacement: PASS")
        print("  nearest-neighbor/Jaccard: PASS")

if __name__=="__main__": main()
