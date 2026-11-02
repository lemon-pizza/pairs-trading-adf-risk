"""Rebuild published cost-adjusted results from the frozen monthly selections."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    commands = [
        [sys.executable, '-c', 'from release.validate import verify_frozen_inputs; verify_frozen_inputs()'],
        [sys.executable, '-m', 'release.costs'],
        [sys.executable, '-m', 'release.validate'],
        [sys.executable, '-m', 'release.report'],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print('Complete: README.md, results/costs, checks, reports')


if __name__ == '__main__':
    main()
