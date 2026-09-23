#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, os, platform, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "dental_matsci_word2vec_reproducibility_package"
EXCLUDED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "dist"}
PRIVATE_PREFIXES = (
    "outputs_1992_2017_xml_pdf", "outputs_after2018_xml_pdf", "outputs_shared",
    "outputs_models", "outputs_procrustes_30000", "outputs_procrustes_50000",
    "outputs_procrustes_methods", "outputs_procrustes_diagnostic", "outputs_semantic_shift",
    "outputs_consistency_90pct",
)
REQUIRED = [
    "README.md", "requirements.txt", "configs/target_terms.tsv", "configs/anchor_terms_consistency.txt",
    "scripts/modeling/build_shared_phraser.py", "scripts/modeling/train_word2vec_from_tokens.py",
    "scripts/analysis/procrustes_displacement.py", "scripts/analysis/procrustes_diagnostic.py",
    "scripts/analysis/calculate_procrustes_summary.py", "scripts/analysis/calculate_procrustes_sensitivity.py",
    "scripts/analysis/semantic_shift_evidence.py", "scripts/robustness/sample90_and_train_word2vec.py",
    "scripts/robustness/compare_word2vec_consistency.py", "validation/run_all_validation.py",
    "examples/build_inspectable_examples.py", "examples/validate_examples.py",
    "examples/manuscript_output_snapshots/cosine_displacement_unique_terms.csv",
    "run_metadata/matsci_pre2018_sharedphrases.config.json", "run_metadata/matsci_post2018_sharedphrases.config.json",
    "reproduction/README.md", "reproduction/EXPECTED_RESULTS.json",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_logged(cmd: list[str], log_path: Path, timeout: int = 240) -> None:
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(p.stdout, encoding="utf-8")
    print(p.stdout, end="")
    if p.returncode:
        raise SystemExit(f"Command failed ({p.returncode}): {' '.join(cmd)}")

def should_exclude(rel: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if rel.parts[:2] == ("reproduction", "validation_logs"):
        return True
    if rel.parts and rel.parts[0] in PRIVATE_PREFIXES:
        return True
    if rel.suffix in {".pyc", ".pyo"}:
        return True
    return False

def git_info() -> dict:
    def run(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain"), "tag_exact": run("describe", "--tags", "--exact-match")}

def main():
    ap = argparse.ArgumentParser(description="Build sanitized checksummed manuscript reproducibility bundle.")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--dist-dir", default="dist")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--allow-missing-generated", action="store_true", help="Development-only: do not require generated inspectable pipeline.")
    args = ap.parse_args()

    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("Missing required repository files:\n  " + "\n  ".join(missing))
    git = git_info()
    if not git["commit"]:
        raise SystemExit(
            "Refusing to build release outside a Git repository."
        )
    
    if git["status_porcelain"]:
        raise SystemExit(
            "Refusing to build release from a dirty Git working tree."
        )
    
    generated_manifest = ROOT / "examples/inspectable_pipeline/MANIFEST.json"
    if not generated_manifest.is_file() and not args.allow_missing_generated:
        raise SystemExit("Generated inspectable pipeline is missing. Run: python3 examples/build_inspectable_examples.py")

    logs_dir = ROOT / "reproduction/validation_logs"
    if not args.skip_validation:
        run_logged([sys.executable, "validation/run_all_validation.py"], logs_dir / "run_all_validation.log")
        ex_cmd = [sys.executable, "examples/validate_examples.py"]
        if not args.allow_missing_generated:
            ex_cmd.append("--require-generated")
        run_logged(ex_cmd, logs_dir / "validate_examples.log")

    subprocess.run([sys.executable, "reproduction/capture_environment.py", "--out-dir", "reproduction/environment"], cwd=ROOT, check=True)

    dist = (ROOT / args.dist_dir).resolve()
    dist.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="repro_bundle_"))
    bundle = work / args.name
    bundle.mkdir()

    copied = 0
    for src in sorted(ROOT.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if should_exclude(rel):
            continue
        dst = bundle / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # Guard against accidentally bundling very large/private files.
    suspicious = []
    for p in bundle.rglob("*"):
        if p.is_file() and p.stat().st_size > 100 * 1024 * 1024:
            suspicious.append(str(p.relative_to(bundle)))
    if suspicious:
        raise SystemExit("Refusing to bundle files >100 MB; inspect for private/full-corpus data:\n  " + "\n  ".join(suspicious))

    build_info = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bundle_name": args.name,
        "source_root": ".",
        "python": sys.version,
        "platform": platform.platform(),
        "git": git,
        "generated_inspectable_pipeline_present": generated_manifest.is_file(),
        "validation_skipped": bool(args.skip_validation),
        "copied_files_before_manifest": copied,
    }
    (bundle / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    entries = []
    for p in sorted(bundle.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(bundle).as_posix()
        if rel in {"FILE_MANIFEST.csv", "SHA256SUMS"}:
            continue
        entries.append((rel, p.stat().st_size, sha256(p)))

    with (bundle / "FILE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_bytes", "sha256"])
        w.writerows(entries)
    (bundle / "SHA256SUMS").write_text("".join(f"{digest}  {rel}\n" for rel, _, digest in entries), encoding="utf-8")

    zip_path = dist / f"{args.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{args.name}/{p.relative_to(bundle).as_posix()}")

    final_dir = dist / args.name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(bundle, final_dir)
    shutil.rmtree(work)

    print("Reproducibility bundle build PASS")
    print(f"  directory: {final_dir}")
    print(f"  zip:       {zip_path}")
    print(f"  manifest entries: {len(entries)}")

if __name__ == "__main__":
    main()
