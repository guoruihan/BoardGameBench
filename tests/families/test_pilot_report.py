import json
from pathlib import Path
import runpy


def test_settled_resource_failure_overrides_raw_completed_game(tmp_path):
    summarize = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/report_harmonies_pilot.py'))['summarize']
    natural = {'status': 'complete', 'natural': True, 'own_score': 100, 'opponent_score': 80, 'win_points': 1.}
    row = {'match_id': 'x', 'profile': 'L20', 'control': False, 'seed': 1, 'seat': 0,
           'submission_sha256': 'a', 'result': {'status': 'cpu_exhausted', 'natural': False, 'win_points': 0.,
           'gt': {'outcomes': [natural], 'sources': {'program': 10, 'default_random': 2}},
           'model': {'usage': {'calls': 2, 'input_tokens': 100, 'output_tokens': 20}},
           'compute': {'cpu_seconds': 121., 'active_wall_seconds': 150.}}}
    (tmp_path/'results.jsonl').write_text(json.dumps(row)+'\n')
    rows, report = summarize(tmp_path)
    assert rows[0]['own_score'] is None and rows[0]['win_points'] == 0.
    group = report['groups']['developed-L20']
    assert group['natural'] == 0 and group['win_denominator'] == 1 and group['win_points_mean'] == 0.
    assert group['calls'] == 2 and group['cpu_seconds'] == 121.
    assert report['paired']['developed-L20_minus_control-L20']['complete_seed_pairs'] == 0
