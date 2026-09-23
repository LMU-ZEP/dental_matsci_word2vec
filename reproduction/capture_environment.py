#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, platform, subprocess, sys
from pathlib import Path

PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("scikit-learn", "sklearn"), ("gensim", "gensim"), ("spacy", "spacy"),
    ("nltk", "nltk"), ("pypdf", "pypdf"), ("PyMuPDF", "fitz"),
    ("umap-learn", "umap"), ("ijson", "ijson"),
]

def pkg_version(import_name: str):
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", None) or getattr(mod, "VERSION", None) or "installed"
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reproduction/environment")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = {
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: pkg_version(import_name) for name, import_name in PACKAGES},
    }
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        info["spacy_model_en_core_web_sm"] = nlp.meta.get("version", "installed")
    except Exception as exc:
        info["spacy_model_en_core_web_sm"] = f"UNAVAILABLE: {type(exc).__name__}"
    try:
        from nltk.corpus import stopwords
        info["nltk_english_stopwords"] = len(stopwords.words("english"))
    except Exception as exc:
        info["nltk_english_stopwords"] = f"UNAVAILABLE: {type(exc).__name__}"

    (out / "environment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], text=True, stdout=subprocess.PIPE, check=False)
    (out / "pip_packages.txt").write_text(freeze.stdout, encoding="utf-8")
    print(f"Environment captured in {out}")

if __name__ == "__main__":
    main()
