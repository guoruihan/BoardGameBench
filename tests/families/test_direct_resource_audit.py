"""Accounting regressions: inclusive token fields and incomplete matches."""
import json
from pathlib import Path
import runpy

import pytest


AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/audit_v175_resources.py'))


@pytest.mark.parametrize('label,usage,expected', [
    ('codex', dict(input_tokens=100, cached_input_tokens=40, output_tokens=30,
                   reasoning_output_tokens=20), (60, 40, 0, 10, 20, 130)),
    ('opencode', dict(input=100, output=10, reasoning=20, cache=dict(read=40, write=0),
                     total=170), (100, 40, 0, 10, 20, 170)),
    ('claude', dict(input_tokens=100, cache_read_input_tokens=40,
                   cache_creation_input_tokens=20, output_tokens=30,
                   output_tokens_details=dict(thinking_tokens=20),
                   iterations=[dict(input_tokens=100, output_tokens=30)],
                   cache_creation=dict(ephemeral_1h_input_tokens=20)),
     (100, 40, 20, 10, 20, 190)),
])
def test_inclusive_fields_and_breakdowns_are_not_counted_twice(label, usage, expected):
    result = AUDIT['normalized_usage'](label, usage)
    keys = ('input_uncached', 'input_cache_read', 'input_cache_write',
            'output_visible', 'output_reasoning', 'total_tokens')
    assert tuple(result[k] for k in keys) == expected


def test_unrecognized_or_inconsistent_usage_fails_closed():
    with pytest.raises(ValueError):
        AUDIT['normalized_usage']('opencode', dict(input=1, total=2))
    with pytest.raises(ValueError):
        AUDIT['normalized_usage']('codex', dict(input_tokens=1, cached_input_tokens=2))
    with pytest.raises(ValueError):
        AUDIT['normalized_usage']('codex', dict(cache_write_input_tokens=1))


def test_missing_response_and_unstarted_match_are_retained(tmp_path):
    job = dict(id='sample', game='harmonies', opponent='legacy', seed=1, seat=1, label='codex')
    directory = tmp_path / 'matches' / job['id']
    directory.mkdir(parents=True)
    (directory / 'result.json').write_text(json.dumps(dict(job=job, status='user_stopped',
        calls=2, seat_decisions=[3, 2], parent_cpu_seconds=5, wall_seconds=20)))
    for n in (1, 2):
        (directory / f'call_{n:04d}.request.json').write_text('{}')
    (directory / 'call_0001.response.json').write_text(json.dumps(dict(ok=True,
        usage=[dict(input_tokens=100, output_tokens=30)], wall_seconds=10)))
    stopped, _, _ = AUDIT['audit_job'](tmp_path, job)
    pending, _, _ = AUDIT['audit_job'](tmp_path, {**job, 'id': 'pending'})
    result = AUDIT['aggregate']([stopped, pending])
    assert result['statuses'] == dict(user_stopped=1, not_started=1)
    assert result['missing_responses'] == 1
    assert result['total_tokens'] == 130
    assert result['wins'] == result['draws'] == result['losses'] == 0
    assert result['parent_cpu_seconds'] == 5
    assert result['agent_decisions'] == 2
    assert result['opponent_decisions'] == 3
