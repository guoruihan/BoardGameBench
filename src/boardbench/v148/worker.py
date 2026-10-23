"""Reuse audited full duo games; MCTS diagnostics and visit targets are archived."""
import argparse
import time

from boardbench.artifacts.store import digest, read_json
from boardbench.v146.search import TerminalSearch
from boardbench.v146.worker import play as base_play, run as base_run
from .search import Search

METHODS = ('model', 'habitat', 'search_habitat', 'search_mcts', 'search_mcts_wide', 'search_mcts_value')


def validate_job(job):
    if job['method'] not in METHODS or job['seat'] not in (0, 1):
        raise ValueError('invalid V148 method/seat')
    ranges = {'calibration': (148900000,149000000), 'train': (148000000,148900000),
              'development': (248000000,249000000)}
    if job['split'] not in ranges or not ranges[job['split']][0] <= job['environment_seed'] < ranges[job['split']][1]:
        raise ValueError('invalid V148 split; formal tests excluded')
    if not 0 < job['seconds'] <= 900 or job['deadline_unix'] <= time.time():
        raise ValueError('expired/unbounded game')
    for name in ('student', 'opponent'):
        if digest(job[name]) != job[name+'_sha256']:
            raise ValueError('immutable policy mismatch')


def play(job, progress=None):
    def factory(focal, opponent, seat, continuation, seconds, seed, **kwargs):
        if continuation == 'habitat':
            return TerminalSearch(focal, opponent, seat, continuation, seconds, seed, **kwargs)
        options = job.get('mcts_config', {})
        if set(options) - {'max_depth', 'initial_width', 'widening', 'exploration', 'value_fraction', 'min_value_depth'}:
            raise ValueError('unknown MCTS option')
        value = None
        if continuation == 'mcts_value':
            from .value import LeafValue
            value = LeafValue(job['value_manifest'], job['value_manifest_sha256'], focal.native,
                              expected_opponent=job['opponent_sha256'])
        return Search(focal, opponent, seat, continuation, seconds, seed,
                      value_evaluator=value, **options, **kwargs)
    result, trace = base_play(job, progress, validator=validate_job, search_factory=factory)
    trace['format'] = 'v148_duo_search_trajectory'
    infos = [r['search'] for r in trace['decisions'] if r['search']]
    for name in ('expanded_nodes', 'terminal_leaves', 'value_leaves'):
        result[name] = sum(i.get(name, 0) for i in infos)
    result['max_tree_depth'] = max((i.get('max_tree_depth', 0) for i in infos), default=0)
    result['max_turn_depth'] = max((i.get('max_turn_depth', 0) for i in infos), default=0)
    result['visited_actions'] = sum(sum(c['samples']>0 for c in i['candidates']) for i in infos)
    return result, trace


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); base_run(read_json(a.job), a.out, play_function=play)
