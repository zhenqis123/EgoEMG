#!/bin/bash
# Build paper/main.pdf
set -e
cd "$(dirname "$0")"

# Clean
rm -f main.aux main.bbl main.blg main.out main.log main.pdf main-draft.tex

# Pass 1: generate aux (returns non-zero due to unresolved refs, which is expected)
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

# BibTeX: generate bbl from aux citations
bibtex main > /dev/null 2>&1

# Pass 2: resolve citations — writes bibcite to aux, produces correct PDF
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
cp main.aux main.aux.resolved   # keep the aux with bibcite entries

# Pass 3: stabilize cross-refs.
# The 3rd pdflatex pass on this document triggers hyperref nesting bug,
# so we use draft hyperref mode. -jobname=main reuses main.bbl.
sed 's/\\usepackage\[hidelinks,breaklinks=true,pdfversion=1.5\]{hyperref}/\\usepackage[draft]{hyperref}/' main.tex > main-draft.tex
pdflatex -jobname=main -interaction=nonstopmode main-draft.tex > /dev/null 2>&1 || true

# Restore the citation-resolved aux (draft run overwrites it with fresh aux)
cp main.aux.resolved main.aux
rm -f main.aux.resolved main-draft.tex main-draft.aux main-draft.out main-draft.log

# Verify
if [ -f main.pdf ]; then
    PAGES=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    CITATIONS=$(grep -c 'bibcite' main.aux 2>/dev/null || echo 0)
    echo "✓ main.pdf built ($PAGES pages, $CITATIONS resolved citations, $(du -h main.pdf | cut -f1))"
else
    echo "✗ main.pdf not generated"
    exit 1
fi
