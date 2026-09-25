#!/usr/bin/env bash
# Build the paper and package the files arXiv needs into arxiv_submission.tar.gz.
# arXiv does not run BibTeX, so the compiled main.bbl is included.
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null

rm -rf arxiv_build arxiv_submission.tar.gz
mkdir -p arxiv_build/figures arxiv_build/tables
cp main.tex main.bbl arxiv_build/
cp tables/*.tex arxiv_build/tables/
# Only the figures the paper actually includes.
for f in $(grep -o 'figures/[A-Za-z0-9_.-]*' main.tex | sort -u); do
  cp "$f" "arxiv_build/$f"
done

# Check that the package compiles on its own, as arXiv will compile it.
(cd arxiv_build && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
rm -f arxiv_build/*.aux arxiv_build/*.log arxiv_build/*.out arxiv_build/*.toc arxiv_build/main.pdf

tar czf arxiv_submission.tar.gz -C arxiv_build .
rm -rf arxiv_build
echo "Wrote $(pwd)/arxiv_submission.tar.gz"
tar tzf arxiv_submission.tar.gz
