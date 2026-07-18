#!/usr/bin/env bash
# Build script for AAAI 2026 submission
set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="build"
mkdir -p "$OUTDIR"

echo "=== Building AAAI 2026 submission (pdflatex + bibtex) ==="

# Pass 1: pdflatex
pdflatex -interaction=nonstopmode \
  -output-directory="$OUTDIR" main.tex > /dev/null 2>&1 || true

# bibtex
bibtex "$OUTDIR/main" 2>&1 | grep -v "^$" | head -20 || true

# Pass 2 & 3: pdflatex (resolve references)
pdflatex -interaction=nonstopmode \
  -output-directory="$OUTDIR" main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode \
  -output-directory="$OUTDIR" main.tex 2>&1 | tail -5

echo ""
if [ -f "$OUTDIR/main.pdf" ]; then
  echo "✓ Build 成功: $OUTDIR/main.pdf ($(du -h $OUTDIR/main.pdf | cut -f1))"
  echo "  Pages: $(pdfinfo $OUTDIR/main.pdf 2>/dev/null | grep Pages || echo '?')"
else
  echo "✗ Build 失败, 检查 $OUTDIR/main.log"
fi
