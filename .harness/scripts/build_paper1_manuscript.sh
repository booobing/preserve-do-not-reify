#!/usr/bin/env bash
# Build papers/preserve_vs_reify/preserve_vs_reify.pdf with the repo's TinyTeX (pdflatex + bibtex).
#   bash .harness/scripts/build_paper1_manuscript.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.harness/tools/TinyTeX/TinyTeX/bin/windows"
cd "$ROOT/papers/preserve_vs_reify"
"$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error preserve_vs_reify.tex > build.log
"$BIN/bibtex.exe" preserve_vs_reify >> build.log
"$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error preserve_vs_reify.tex >> build.log
"$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error preserve_vs_reify.tex >> build.log
grep -E "Warning: (Citation|Reference)|undefined|Overfull" build.log | sort | uniq -c || true
echo "built $(pwd)/preserve_vs_reify.pdf"
