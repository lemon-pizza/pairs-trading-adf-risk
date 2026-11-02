"""Rerun the original full-graph selector without replacing published results.

The frozen selections are the reproducible published experiment. This optional
entry point rebuilds candidates and repeats the bounded search in work/rerun.
Wall-clock search budgets can produce different feasible selections on another
machine. The original trading and selection kernels are imported unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/adf_risk_selection'
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT / 'src'))
_spec = importlib.util.spec_from_file_location('_preserved_adf_run', STUDY / 'run.py')
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def writable_output(path):
    """Allow generated results only below the checkout's work directory."""
    resolved = Path(path).resolve()
    work = (ROOT / 'work').resolve()
    if resolved == work or not resolved.is_relative_to(work):
        raise ValueError('Optimization output must be a subdirectory of work/.')
    return resolved


def verify_frozen_inputs():
    manifest = json.loads((ROOT / 'checks/frozen_inputs.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or runner.digest(path) != expected:
            raise ValueError('Frozen input differs: ' + name)
    return len(manifest['files'])


def one_month(key, end, output, settings):
    return runner.month(key, end, output, settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'work/rerun')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--months', type=int, help='Explicit partial run, starting with the first month')
    parser.add_argument('--time-limit', type=float, default=8., help='Seconds per degree cap; published experiment used 8')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.time_limit <= 0 or (args.months is not None and args.months < 1):
        parser.error('workers, months, and time-limit must be positive')
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, '-m', 'release.optimize', *sys.argv[1:]])
    try:
        output = writable_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    protected = verify_frozen_inputs()
    output.mkdir(parents=True, exist_ok=True)
    all_keys = sorted(p.stem for p in (ROOT / 'data/selection').glob('*.parquet'))
    if len(all_keys) != 77:
        raise ValueError('Expected the 77 published formation dates')
    ends = dict(zip(all_keys, all_keys[1:] + ['20230531']))
    keys = all_keys if args.months is None else all_keys[:args.months]
    settings = {'train_days': 252, 'time_limit': args.time_limit,
                'structural_milp': False, 'max_moves': 1000}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(one_month, key, ends[key], str(output), settings) for key in keys]
        for count, future in enumerate(as_completed(pending), 1):
            print(json.dumps(runner.clean({'completed': count, 'total': len(keys), **future.result()})), flush=True)
    complete = keys == all_keys
    metrics = runner.analyze(output, keys, complete)
    verify_frozen_inputs()
    runner.dump(output / 'execution.json', {
        'months': keys, 'complete': complete, 'settings': settings,
        'code': runner.code_identity(), 'metrics': metrics,
        'protected_frozen_files': protected, 'frozen_inputs_unchanged': True,
        'costs': 'Gross only; costs are separately evaluated for the frozen published selections.',
        'interpretation': 'New bounded-search realization; not a replacement for the frozen published experiment.'})
    print(json.dumps({'complete': complete, 'output': str(output.relative_to(ROOT)),
                      'months': len(keys), 'frozen_inputs_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
