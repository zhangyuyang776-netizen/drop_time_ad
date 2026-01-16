#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mpiexec -n 2 pytest -q -s tests/test_p3_5_5_snes_smoke_additive_vs_schur_mpi.py -rs
mpiexec -n 2 pytest -q -s tests/test_p3_4_5_regression_fieldsplit_shell_safe_mpi.py -rs
mpiexec -n 2 pytest -q -s tests/test_p3_4_1_5_failfast_fieldsplit_use_amat_options_mpi.py -rs
