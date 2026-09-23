#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ["run_smoke_test.py","run_mini_e2e.py","run_regression_test.py"]:
    print("\n"+"="*80+f"\n{name}\n"+"="*80,flush=True)
    try:
        p=subprocess.run([sys.executable,str(ROOT/"validation"/name)],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    except subprocess.TimeoutExpired as e:
        print(e.stdout or "")
        raise SystemExit(f"{name} timed out")
    print(p.stdout,end="")
    if p.returncode:
        raise SystemExit(p.returncode)
print("\nALL VALIDATION TESTS PASSED")
