"""Run the original finite-graph tests without overwriting frozen evidence."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = ROOT / 'work'
    work.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONHASHSEED'] = '0'
    with TemporaryDirectory(prefix='optimizer-check-', dir=work) as directory:
        folder = Path(directory)
        for name in ('optimizer.py', 'check_optimizer.py'):
            shutil.copyfile(ROOT / 'research/adf_risk_selection' / name, folder / name)
        subprocess.run([sys.executable, str(folder / 'check_optimizer.py')],
                       cwd=folder, env=env, check=True)
        target = ROOT / 'checks/optimizer_validation.json'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(folder / 'optimizer_checks.json', target)


if __name__ == '__main__':
    main()
