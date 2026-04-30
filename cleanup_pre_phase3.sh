#!/bin/bash
# ============================================================
# StochastiQ - Pre-Phase-3 Cleanup
# Handles polish items before moving into stochastic model calibration:
#   1. Removes redundant simple_returns.parquet (always derived from log returns)
#   2. Removes redundant .gitkeep files (folders are no longer empty)
#   3. Installs src/config.py with project-wide constants
# ============================================================

set -e

PROJECT_DIR="$HOME/Downloads/StochastiQ"

echo "======================================"
echo "  StochastiQ Pre-Phase-3 Cleanup"
echo "======================================"
echo ""

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project not found at $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

# ============================================================
# 1. Remove the redundant simple_returns dataset
# ============================================================
echo "[1/3] Removing redundant simple_returns.parquet..."
if [ -f "data/processed/simple_returns.parquet" ]; then
  rm "data/processed/simple_returns.parquet"
  echo "      Removed. Simple returns will be derived from log returns when needed."
else
  echo "      Already absent. Nothing to do."
fi

# ============================================================
# 2. Remove redundant .gitkeep files
# ============================================================
echo "[2/3] Removing redundant .gitkeep files..."
removed_count=0
for gitkeep in "data/processed/.gitkeep" "reports/figures/.gitkeep"; do
  if [ -f "$gitkeep" ]; then
    rm "$gitkeep"
    echo "      Removed: $gitkeep"
    removed_count=$((removed_count + 1))
  fi
done
if [ $removed_count -eq 0 ]; then
  echo "      None found."
fi

# Note: data/raw/.gitkeep and reports/tables/.gitkeep are kept because
# those directories are still empty and we want them tracked in git.

# ============================================================
# 3. Install the project-wide config module
# ============================================================
echo "[3/3] Installing src/config.py..."

cat > src/config.py << 'EOF'
"""
Project-wide configuration constants for StochastiQ.

Centralizing these here ensures every notebook and module uses the same
values for the risk-free rate, trading day convention, and other
parameters that appear in multiple phases of the project.
"""

from __future__ import annotations

# ============================================================
# Market conventions
# ============================================================

# Annualized risk-free rate.
# Set to 4.0%, reflecting the prevailing short-term Treasury environment
# at the time of analysis (Q1 2026). Used for Sharpe/Sortino ratios,
# Black-Scholes option pricing, and stochastic-rate model anchoring.
RISK_FREE_RATE: float = 0.04

# Trading days per year. The standard US equity convention.
TRADING_DAYS: int = 252

# ============================================================
# Risk metrics
# ============================================================

# Default confidence level for VaR and CVaR computations.
DEFAULT_CONFIDENCE: float = 0.95

# ============================================================
# Simulation defaults
# ============================================================

# Default number of Monte Carlo paths for forward simulation.
# 5,000 paths gives a stable distribution estimate for portfolio statistics
# without being prohibitively slow on a laptop.
DEFAULT_N_PATHS: int = 5_000

# Default forward simulation horizon (one trading year).
DEFAULT_HORIZON_DAYS: int = 252

# Random seed for reproducibility across simulation runs.
RANDOM_SEED: int = 42
EOF

echo "      Installed at src/config.py"

# Verify it imports cleanly
python -c "from src.config import RISK_FREE_RATE, TRADING_DAYS; print(f'      Verified: RISK_FREE_RATE={RISK_FREE_RATE}, TRADING_DAYS={TRADING_DAYS}')" 2>/dev/null || {
  echo "      WARNING: config.py installed but import failed. Are you running from inside the venv?"
  echo "      Run 'source venv/bin/activate' first if needed."
}

echo ""
echo "======================================"
echo "  Cleanup complete"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Re-run Notebook 01 to regenerate clean outputs:"
echo "       Open Jupyter Lab, restart kernel, Run All Cells on 01_data_pull.ipynb"
echo "       (Note: Notebook 01's save step will skip simple_returns.parquet"
echo "        from now on after the small edit we'll make next.)"
echo ""
echo "  2. Commit the changes:"
echo "       git add ."
echo "       git commit -m \"Pre-Phase-3 cleanup: add config module, remove redundant files\""
echo "       git push"
echo ""
