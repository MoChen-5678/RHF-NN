#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -m dpl_rhf train-rhf --model PKA1 --z 8 --n 8 \
  --epochs 500 --device auto --graph-offload memory \
  --out "$ROOT/outputs/PKA1/O16" "$@"
