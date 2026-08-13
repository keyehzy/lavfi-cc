#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

if ! command -v "${LAVFI_CC_CLANG:-clang}" >/dev/null 2>&1; then
  echo "Week 4 requires Clang (set LAVFI_CC_CLANG to its path)" >&2
  exit 1
fi

python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/benchmark-week4.py
