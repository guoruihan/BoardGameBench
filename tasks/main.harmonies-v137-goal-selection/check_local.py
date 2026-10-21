"""Run source-matched regression and retain all prior validation costs."""
from pathlib import Path
import subprocess
import sys
import time

from boardbench.artifacts.store import read_json, source_id, write_json

root = Path('outputs/v137/selection_a226_20261022_b')
prior = Path('outputs/v137/selection_a226_20261022_a/local_checks.json')
attempt = len(list(root.glob('local_attempt_*.json'))) + 1
source = source_id()
started = time.monotonic()
tests = sys.argv[1:] or [f'tests/test_v{version}.py' for version in (127,129,133,134,135,136,137)]
with (root / f'local_attempt_{attempt:02d}.log').open('w') as log:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests],
        stdout=log, stderr=subprocess.STDOUT)
receipt = {'source_id': source, 'passed': result.returncode == 0 and source == source_id(),
           'returncode': result.returncode, 'tests':tests, 'cpu_core_seconds': 4*(time.monotonic()-started),
           'ended_unix': time.time()}
write_json(root / f'local_attempt_{attempt:02d}.json', receipt)
receipts = [read_json(p) for p in root.glob('local_attempt_*.json')]
write_json(root / 'local_checks.json', {**receipt, 'prior_receipt': str(prior),
    'cpu_core_seconds': read_json(prior)['cpu_core_seconds'] + sum(r['cpu_core_seconds'] for r in receipts),
    'attempts_this_run': len(receipts)})
raise SystemExit(0 if receipt['passed'] else 1)
