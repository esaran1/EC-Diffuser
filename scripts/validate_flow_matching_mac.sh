#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "Python executable: $(command -v python)"
python --version
PYTHONPATH="${repo_root}/diffuser" python -c 'import torch; print("PyTorch version: {}".format(torch.__version__))'

PYTHONPATH="${repo_root}/diffuser" python -m pytest -q tests/test_flow_matching.py
PYTHONPATH="${repo_root}/diffuser" python diffuser/scripts/smoke_flow_matching.py
PYTHONPATH="${repo_root}/diffuser" python diffuser/scripts/overfit_flow_matching.py
PYTHONPATH="${repo_root}/diffuser" python diffuser/scripts/validate_flow_config.py

echo "MAC FLOW-MATCHING VALIDATION: PASS"
