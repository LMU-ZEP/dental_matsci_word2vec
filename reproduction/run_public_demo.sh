#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 examples/build_inspectable_examples.py
python3 examples/validate_examples.py --require-generated
