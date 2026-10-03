"""Fixed-resource timed evaluation, retaining every attempt including failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import statistics

from boardbench.artifacts.store import read_json, write_json
from .timed import aggregate, run_timed


def episode(job):
    task, spec, seed, seconds = job
    return run_timed(task, spec, seed, seconds)


def evaluate(task, spec, seeds, seconds, out, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', {'task': task, 'policy': spec, 'seeds': seeds,
                                    'seconds': seconds, 'workers': workers, 'threads_per_policy': 1})
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = ((task, spec, seed, seconds) for seed in seeds)
        with (out / 'episodes.jsonl').open('x') as stream:
            for result in pool.map(episode, jobs):
                rows.append(result)
                stream.write(json.dumps(result, separators=(',', ':')) + '\n'); stream.flush()
    assert [r['seed'] for r in rows] == seeds
    summary = aggregate(rows)
    write_json(out / 'summary.json', summary)
    return summary


def paired(candidate, incumbent):
    a, b = ({r['seed']: r for r in rows} for rows in (candidate, incumbent))
    if set(a) != set(b) or len(a) < 2 or len(a) != len(candidate) or len(b) != len(incumbent):
        raise ValueError('paired evaluation requires matching unique seeds')
    diffs = [a[s]['score'] - b[s]['score'] for s in sorted(a)]
    mean = statistics.mean(diffs)
    half = 1.96 * statistics.stdev(diffs) / len(diffs)**.5
    return {'difference': mean, 'normal_95_interval': [mean - half, mean + half], 'n': len(diffs),
            'scope': 'fixed artifacts/environment pairs; selection gate, not independent test evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    evaluate(config['task'], config['policy'], config['seeds'], config['seconds'], args.out,
             config.get('workers', 4))


if __name__ == '__main__':
    main()
