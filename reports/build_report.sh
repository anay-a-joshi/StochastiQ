#!/usr/bin/env bash
# Compile the StochastiQ MGT 6081-A final report.
#
# Usage:  cd reports/ && bash build_report.sh
#
# Requires: pdflatex and bibtex (MacTeX or BasicTeX).
# Install (one time): brew install --cask mactex-no-gui
#
# Output: StochastiQ_Final_Report.pdf in this directory.

set -euo pipefail

cd "$(dirname "$0")"

# Sanity check
if ! command -v pdflatex >/dev/null 2>&1; then
    echo "ERROR: pdflatex not found. Install MacTeX first:" >&2
    echo "    brew install --cask mactex-no-gui" >&2
    echo "    (or install TeX Live: https://www.tug.org/texlive/)" >&2
    exit 1
fi

if [ ! -f references.bib ]; then
    echo "ERROR: references.bib not found. It must live next to the .tex file." >&2
    exit 1
fi

# Check that figure directory exists with at least the key figures
MISSING_FIGS=0
for f in figures/01_correlation_matrix.png \
         figures/01_rolling_volatility.png \
         figures/02_drawdowns.png \
         figures/02_efficient_frontier.png \
         figures/03_qq_plots.png \
         figures/04_simulated_correlations.png \
         figures/04_per_model_weights.png \
         figures/04_cross_evaluation.png \
         figures/05_payoff_diagrams.png \
         figures/05_greeks_comparison.png \
         figures/05_sharpe_grid.png \
         figures/06_sharpe_diff_bootstrap.png \
         figures/06_predicted_vs_realized.png \
         figures/07_regime_timeline.png \
         figures/07_regime_validation.png \
         figures/07_sharpe_forest_plot.png \
         figures/07_cvar_by_regime.png \
         figures/08_sharpe_vs_cvar_divergence.png; do
    if [ ! -f "$f" ]; then
        echo "WARNING: missing figure: $f" >&2
        MISSING_FIGS=$((MISSING_FIGS+1))
    fi
done

if [ $MISSING_FIGS -gt 0 ]; then
    echo "" >&2
    echo "WARNING: $MISSING_FIGS figure(s) missing. PDF will compile but show error boxes." >&2
    echo "Make sure all figures from Phases 1-8 are in reports/figures/." >&2
    echo "" >&2
fi

echo "Pass 1/4: pdflatex (initial layout)..."
pdflatex -interaction=nonstopmode StochastiQ_Final_Report.tex > /dev/null

echo "Pass 2/4: bibtex (resolve citations)..."
bibtex StochastiQ_Final_Report > /dev/null

echo "Pass 3/4: pdflatex (incorporate bibliography)..."
pdflatex -interaction=nonstopmode StochastiQ_Final_Report.tex > /dev/null

echo "Pass 4/4: pdflatex (final cross-references)..."
pdflatex -interaction=nonstopmode StochastiQ_Final_Report.tex > /dev/null

# Clean up auxiliary files but keep the PDF
rm -f StochastiQ_Final_Report.aux \
      StochastiQ_Final_Report.log \
      StochastiQ_Final_Report.out \
      StochastiQ_Final_Report.bbl \
      StochastiQ_Final_Report.blg \
      StochastiQ_Final_Report.toc

if [ -f StochastiQ_Final_Report.pdf ]; then
    pages=$(pdfinfo StochastiQ_Final_Report.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
    size_kb=$(du -k StochastiQ_Final_Report.pdf | cut -f1)
    echo ""
    echo "============================================================"
    echo "*** Build successful ***"
    echo "    Output:  $(pwd)/StochastiQ_Final_Report.pdf"
    echo "    Pages:   ${pages:-?}"
    echo "    Size:    ${size_kb} KB"
    echo "============================================================"
    echo ""
    echo "To open: open StochastiQ_Final_Report.pdf"
else
    echo "ERROR: PDF was not produced. Re-run pdflatex manually to see the error:" >&2
    echo "    pdflatex StochastiQ_Final_Report.tex" >&2
    exit 1
fi
