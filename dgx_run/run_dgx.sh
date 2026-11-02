#!/usr/bin/env bash
# Start the full benchmark in the background. It keeps running after you close the
# terminal or log out; progress goes to a timestamped log file in this folder.
#
#   bash run_dgx.sh                    every free GPU, full run
#   bash run_dgx.sh --check            only show which Python and dataset would be used
#   bash run_dgx.sh --smoke            short end-to-end test on 3,000 windows (do this first)
#   bash run_dgx.sh --gpus 0,1,2,3     only these GPUs
#   PYTHON=/path/to/python bash run_dgx.sh     force a specific interpreter
#
# The Python interpreter is found automatically, preferring an environment named
# project_venv (virtualenv or conda); the first interpreter with every required package wins.
set -eo pipefail
cd "$(dirname "$0")"
ME="${USER:-$(whoami)}"
# StratifiedGroupKFold needs scikit-learn >= 1.0 and binomtest needs SciPy >= 1.7
# Same import order and safeguard as rigorous_benchmark.py: block the optional torch_scatter
# binary (a mismatched build aborts the interpreter) and import XGBoost before PyTorch.
REQ="import sys; sys.modules['torch_scatter'] = None; sys.modules['torch_sparse'] = None; import xgboost, torch, torch_geometric, netCDF4, pandas, joblib; from sklearn.model_selection import StratifiedGroupKFold; from scipy.stats import binomtest"
LIGHT_PKGS="torch_geometric xgboost netCDF4 pandas joblib scikit-learn>=1.0 scipy>=1.7"

CANDIDATES=()
VENVS=()
add() { [ -n "$1" ] && [ -x "$1" ] && CANDIDATES+=("$1"); return 0; }

# 1. explicit choice or an already activated environment
add "${PYTHON:-}"
[ -n "${VIRTUAL_ENV:-}" ] && add "$VIRTUAL_ENV/bin/python"
[ -n "${CONDA_PREFIX:-}" ] && add "$CONDA_PREFIX/bin/python"

# 2. project_venv in the usual places, then anywhere a few levels below home
for d in "$PWD/project_venv" "$PWD/../project_venv" "$HOME/project_venv" "$HOME/PROJECT/project_venv" \
         "/jupyter/$ME/project_venv" "/jupyter/$ME/PROJECT/project_venv"; do
  [ -x "$d/bin/python" ] && VENVS+=("$d/bin/python")
done
while IFS= read -r d; do
  [ -x "$d/bin/python" ] && VENVS+=("$d/bin/python")
done < <(find "$HOME" "/jupyter/$ME" -maxdepth 4 -type d -name project_venv 2>/dev/null)
if command -v conda >/dev/null 2>&1; then
  p=$(conda env list 2>/dev/null | awk '$1=="project_venv"{print $NF}')
  [ -n "$p" ] && [ -x "$p/bin/python" ] && VENVS+=("$p/bin/python")
fi
for v in "${VENVS[@]}"; do add "$v"; done

# 3. other project environments and the system interpreters
for d in "$HOME/PROJECT/venv" "/jupyter/$ME/PROJECT/venv" "$HOME/venv"; do add "$d/bin/python"; done
add "$(command -v python3 || true)"
add "$(command -v python || true)"

PY=""
for p in "${CANDIDATES[@]}"; do
  if "$p" -c "$REQ" >/dev/null 2>&1; then PY="$p"; break; fi
done

# 4. project_venv has PyTorch but lacks lighter packages: install them into it
if [ -z "$PY" ] && [ "${#VENVS[@]}" -gt 0 ]; then
  V="${VENVS[0]}"
  if "$V" -c "import torch" >/dev/null 2>&1; then
    echo "Installing missing packages into $V ..."
    "$V" -m pip install --quiet --upgrade-strategy only-if-needed $LIGHT_PKGS && "$V" -c "$REQ" >/dev/null 2>&1 && PY="$V"
  fi
fi

if [ -z "$PY" ]; then
  echo "No Python environment with all required packages was found."
  echo "Checked:"; printf '  %s\n' "${CANDIDATES[@]}"
  echo "Fix: activate your environment (source /path/to/project_venv/bin/activate) and run"
  echo "     python -m pip install torch_geometric xgboost netCDF4 pandas joblib 'scikit-learn>=1.0' 'scipy>=1.7'"
  echo "     (PyTorch itself must already be installed), then run: bash run_dgx.sh $*"
  exit 1
fi

echo "Python: $PY"
"$PY" -c "import sys, xgboost, torch; print('  version', sys.version.split()[0], '| torch', torch.__version__, '| CUDA GPUs', torch.cuda.device_count())"

if [ "${1:-}" = "--check" ]; then          # report only; start nothing
  "$PY" -c "import rigorous_benchmark as rb; print('Dataset:', rb.find_dataset() or 'NOT FOUND (pass --data PATH)')"
  exit 0
fi

LOG="run_$(date +%Y%m%d_%H%M%S).log"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$PY" rigorous_benchmark.py all "$@" > "$LOG" 2>&1 < /dev/null &
else
  nohup "$PY" rigorous_benchmark.py all "$@" > "$LOG" 2>&1 < /dev/null &
fi
echo "Started in the background (PID $!). You can log out now."
echo "  Follow progress:  tail -f $(pwd)/$LOG"
echo "  Check status:     $PY rigorous_benchmark.py status $*"
echo "  Stop everything:  pkill -f rigorous_benchmark.py   (run 'bash run_dgx.sh $*' again to resume)"
