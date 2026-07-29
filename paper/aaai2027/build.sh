#!/usr/bin/env bash
# Build script for AAAI 2027 submission
set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="build"
mkdir -p "$OUTDIR"

echo "=== Building AAAI 2027 submission (pdflatex + bibtex) ==="

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

echo ""
echo "=== Building 9-page main submission ==="

pdflatex -interaction=nonstopmode -jobname=main_submission \
  '\def\mainonly{1}\input{main.tex}' > /dev/null
bibtex main_submission 2>&1 | grep -v "^$" | head -20 || true
pdflatex -interaction=nonstopmode -jobname=main_submission \
  '\def\mainonly{1}\input{main.tex}' > /dev/null
pdflatex -interaction=nonstopmode -jobname=main_submission \
  '\def\mainonly{1}\input{main.tex}' > /dev/null

if [ -f main_submission.pdf ]; then
  echo "✓ Build 成功: main_submission.pdf ($(du -h main_submission.pdf | cut -f1))"
  echo "  Pages: $(pdfinfo main_submission.pdf 2>/dev/null | grep Pages || echo '?')"
else
  echo "✗ Build 失败, 检查 main_submission.log"
  exit 1
fi

echo ""
echo "=== Extracting standalone supplementary material ==="

FULL_PAGES=$(pdfinfo "$OUTDIR/main.pdf" | awk '/^Pages:/ {print $2}')
MAIN_PAGES=$(pdfinfo main_submission.pdf | awk '/^Pages:/ {print $2}')
SUPP_FIRST_PAGE=$((MAIN_PAGES + 1))

if (( SUPP_FIRST_PAGE > FULL_PAGES )); then
  echo "✗ Supplement extraction failed: full PDF has no pages after the main paper"
  exit 1
fi

SUPP_TMP=$(mktemp -d)
trap 'rm -rf "$SUPP_TMP"' EXIT
pdfseparate -f "$SUPP_FIRST_PAGE" -l "$FULL_PAGES" \
  "$OUTDIR/main.pdf" "$SUPP_TMP/page-%d.pdf"

SUPP_PAGES=()
for ((page = SUPP_FIRST_PAGE; page <= FULL_PAGES; page++)); do
  SUPP_PAGES+=("$SUPP_TMP/page-$page.pdf")
done
pdfunite "${SUPP_PAGES[@]}" supplementary.pdf

echo "✓ Build 成功: supplementary.pdf ($(du -h supplementary.pdf | cut -f1))"
echo "  Pages: $(pdfinfo supplementary.pdf 2>/dev/null | grep Pages || echo '?')"
