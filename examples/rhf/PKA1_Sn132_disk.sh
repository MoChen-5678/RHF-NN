#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -m dpl_rhf train-rhf --model PKA1 --z 50 --n 82 \
  --epochs 500 --device auto --graph-offload disk \
  --out "$ROOT/outputs/PKA1/Sn132" "$@"
