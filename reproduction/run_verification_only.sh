#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 validation/run_smoke_test.py
python3 validation/run_regression_test.py
python3 examples/validate_examples.py --require-generated
