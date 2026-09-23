#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Verify SHA256SUMS in an unpacked reproducibility bundle.")
    ap.add_argument("bundle", nargs="?", default=".")
    args = ap.parse_args()
    root = Path(args.bundle).resolve()
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"Missing {sums}")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"Missing file listed in checksum manifest: {rel}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch: {rel}\nexpected {expected}\nactual   {actual}")
        checked += 1
    print(f"Bundle checksum verification PASS ({checked} files)")

if __name__ == "__main__":
    main()
