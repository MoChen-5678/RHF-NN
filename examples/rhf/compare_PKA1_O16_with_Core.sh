#!/usr/bin/env bash
set -euo pipefail
if (($# != 1)); then
  echo "Usage: $0 /absolute/path/to/Core-1204" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -m dpl_rhf compare-core --kind rhf --model PKA1 --z 8 --n 8 \
  --result "$ROOT/outputs/PKA1/O16" --core-dir "$1" \
  --out "$ROOT/outputs/PKA1/O16/core"
