import json
import os
from pathlib import Path

import pytest

from boardbench.families.evaluate import FORMAT, canonical_hash
from boardbench.families.report import build_report, main, union_seconds, training_statuses


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def launch(root, name='ppo_r1', start=3600, cpus=(0, 1), gpu=0, host='a226', kind='ppo'):
    p = root/'calico'/name
    value = dict(host=host, game='calico', kind=kind, cpu_ids=list(cpus), launched_unix=start,
                 output=str(p), command=['https://private.example.invalid/token', 'secret-value'],
                 private_ip='192.0.2.99', auth_token='secret-value')
    if gpu is not None:
        value['gpu'] = gpu
    write(p/'launch.json', value)
    return p


def exit_at(path, value, when):
    (path/'exit_code').write_text(str(value)+'\n')
    os.utime(path/'exit_code', (when, when))


def frozen(directory):
    schedule = [dict(match_id=f'{seed}-{orientation}', pair=['a', 'b'], seed=seed,
                     orientation=orientation, first_player=0, seat_names=['a', 'b'] if not orientation else ['b', 'a'])
                for seed in (10, 11) for orientation in (0, 1)]
    manifest = dict(format=FORMAT, game='calico', phase='dev', config='official', rules_version='1',
                    bots={n: dict(spec=dict(kind='heuristic', path='/private/model.pt'),
                                  identity_sha256=n*64, model=None) for n in ('a', 'b')},
                    schedule=schedule)
    manifest['manifest_sha256'] = canonical_hash(manifest)
    write(directory/'manifest.json', manifest)
    return manifest


def attempt(directory, match, status='complete', rewards=(1, -1), n=1):
    filename = f'{match}_{n}.json'
    result = dict(status=status, rewards=list(rewards) if rewards else None, seconds=1,
                  outcome=dict(scores=[10, 8]), traceback='private connection secret-value')
    write(directory/'attempts'/filename, dict(match_id=match, attempt=n, filename=filename, result=result))


def test_training_counts_actual_rows_and_never_projects_from_updates(tmp_path):
    root = tmp_path/'bots'; p = launch(root)
    write(p/'state.json', dict(phase='training', next_update=100, updated_unix=7000))
    write(p/'rollout_0000.json', [dict(status='complete'), dict(status='truncated')])
    write(p/'rollout_0001.json', [dict(status='complete')])
    write(p/'gate_0007.json', [dict(status='error')])
    write(p/'config.json', dict(games=512, updates=100))
    write(p/'confirmation_summary.json', dict(completed=100000))
    write(p/'best.json', dict(secret='must not import policy metadata'))
    t = build_report(root, now=8000, bootstrap_samples=10)['tasks'][0]
    assert t['status'] == 'activity_unverified'
    assert t['progress']['completed_updates'] == 100
    assert t['written_training']['recorded'] == 4
    assert t['written_training']['completed'] == 2
    assert t['written_training']['stages'] == dict(gate=1, rollout=3)
    assert t['resources']['allocated_gpu_count'] == 1  # GPU zero is a real allocation.
    assert t['process_liveness'] == 'unverified'


def test_cem_progress_and_partial_file_are_not_silently_counted(tmp_path):
    root = tmp_path/'bots'; p = launch(root, name='cem_r1', gpu=None, kind='cem')
    write(p/'state.json', dict(phase='complete', next_generation=4))
    write(p/'g000_population.json', [dict(status='complete'), dict(status='error')])
    (p/'g001_population.json').write_text('[{"secret":"private.example.invalid"')
    exit_at(p, 0, 7200)
    t = build_report(root, now=8000)['tasks'][0]
    assert t['status'] == 'abnormal'
    assert t['progress']['completed_generations'] == 4
    assert t['written_training']['recorded'] == 2
    assert t['resources']['allocated_gpu_card_hours_upper_bound'] == 0
    assert {'code': 'unreadable_or_invalid_json', 'file': 'g001_population.json'} in t['issues']
    assert 'private.example.invalid' not in json.dumps(t)


def test_resource_intervals_exit_mtime_and_shared_gpu_union(tmp_path):
    root = tmp_path/'bots'; a = launch(root)
    write(a/'state.json', dict(phase='complete', next_update=1)); exit_at(a, 0, 7200)
    b = launch(root, name='ppo_r2', start=5400, cpus=(1, 2))
    write(b/'state.json', dict(phase='training', next_update=1))
    report = build_report(root, now=10800)
    r = report['resources']
    assert r['task_allocated_cpu_core_hours_upper_bound'] == 5
    assert r['unique_allocated_cpu_core_hours_upper_bound'] == 4.5
    assert r['task_allocated_gpu_card_hours_upper_bound'] == 2.5
    assert r['unique_allocated_gpu_card_hours_upper_bound'] == 2
    assert report['tasks'][0]['resources']['endpoint'] == 'exit_file_mtime'
    assert report['tasks'][1]['resources']['endpoint'] == 'report_time'
    assert union_seconds([(1, 4), (2, 3), (3, 7), (10, 11)]) == 7


def test_public_allowlist_and_local_alias_filter(tmp_path):
    root = tmp_path/'bots'; launch(root)
    launch(root, name='ppo_ip', host='192.0.2.99')
    public = json.dumps(build_report(root, now=8000))
    for secret in ('a226', '192.0.2.99', 'private.example.invalid', 'secret-value', '"command":', str(tmp_path)):
        assert secret not in public
    local = build_report(root, now=8000, include_host_aliases=True)
    assert [t.get('host_alias') for t in local['tasks']] == [None, 'a226']
    assert '192.0.2.99' not in json.dumps(local)


def test_frozen_league_full_denominator_retry_and_paired_ci(tmp_path):
    root = tmp_path/'bots'; p = root/'calico'/'league_r1'; frozen(p)
    attempt(p, '10-0', status='error', rewards=None)
    attempt(p, '10-0', n=2)
    attempt(p, '10-1', rewards=(0, 0))
    attempt(p, '11-0', status='truncated', rewards=None)
    write(p/'invocation_1.json', {}); write(p/'invocation_2.json', {})
    write(p/'summary.json', dict(manifest_sha256='outdated', completed=4, attempts=3))
    result = build_report(root, now=8000, bootstrap_samples=100)['tasks'][0]['league']
    assert result['scheduled'] == 4 and result['completed'] == 2
    assert result['written']['recorded_attempts'] == 4
    assert result['written']['unique_matches_with_attempts'] == 3
    assert result['written']['not_started'] == 1
    assert result['invocations'] == 2 and not result['saved_summary_count_check']
    a = result['pair_reports'][0]['policies']['a']
    assert a['wins'] == 1 and a['draws'] == 1 and a['losses'] == 0
    assert a['all_scheduled_win_rate'] == .375 and a['completed_win_rate'] == .75
    assert a['all_scheduled_win_rate_bounds'] == [.375, .875]
    assert a['bootstrap']['independent_seeds'] == 2
    assert len(a['bootstrap']['all_scheduled_win_rate_ci95']) == 2
    assert 'private' not in json.dumps(result)
    assert not result['selection_performed']


def test_eval_job_link_and_clean_exit_with_incomplete_schedule(tmp_path):
    root = tmp_path/'bots'; p = root/'calico'/'league_r1'; frozen(p)
    job = launch(root, name='league_r1_job', kind='eval', gpu=None)
    value = json.loads((job/'launch.json').read_text()); value['output'] = '/old/copied/location/league_r1'
    write(job/'launch.json', value); exit_at(job, 0, 7200)
    report = build_report(root, now=8000, bootstrap_samples=10)
    assert len(report['tasks']) == 1
    t = report['tasks'][0]
    assert t['data_run'] == 'calico/league_r1'
    assert t['status'] == 'abnormal'
    assert t['league']['scheduled'] == 4 and t['league']['written']['recorded_attempts'] == 0
    assert dict(code='evaluation_incomplete_after_exit') in t['issues']


def test_manifest_tampering_invalid_reward_and_bad_state_are_flagged(tmp_path):
    root = tmp_path/'bots'; p = root/'calico'/'league_r1'; m = frozen(p)
    m['phase'] = 'confirmation'; write(p/'manifest.json', m)
    t = build_report(root, now=8000, bootstrap_samples=10)['tasks'][0]
    assert t['status'] == 'abnormal' and t['league'] is None
    frozen(p); attempt(p, '10-0', rewards=(1, 1))
    t = build_report(root, now=8000, bootstrap_samples=10)['tasks'][0]
    assert t['status'] == 'abnormal' and t['league']['pair_reports'] is None
    write(p/'state.json', dict(phase=['bad']))
    assert build_report(root, now=8000, bootstrap_samples=10)['tasks'][0]['status'] == 'abnormal'


def test_status_evidence_exit_failure_deadline_and_missing_resources(tmp_path):
    root = tmp_path/'bots'; a = launch(root, name='failed')
    exit_at(a, 1, 7200)
    b = launch(root, name='overdue'); value = json.loads((b/'launch.json').read_text())
    value.update(deadline=7500); value.pop('cpu_ids'); value.pop('gpu'); write(b/'launch.json', value)
    report = build_report(root, now=8000)
    assert report['task_statuses'] == dict(abnormal=2)
    r = report['resources']; assert r['tasks_missing_cpu_interval'] == 1 and r['tasks_missing_gpu_interval'] == 1
    assert report['tasks'][1]['resources']['allocated_cpu_core_hours_upper_bound'] is None
    assert dict(code='deadline_passed_without_exit_record') in report['tasks'][1]['issues']


def test_read_only_cli_new_report_and_no_overwrite_or_output_in_bots(tmp_path):
    root = tmp_path/'bots'; launch(root)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    out = tmp_path/'reports'/'now.json'
    main(['--root', str(root), '--output', str(out), '--bootstrap-samples', '10'])
    assert json.loads(out.read_text())['read_only']
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    assert before == after
    with pytest.raises(SystemExit):
        main(['--root', str(root), '--output', str(out)])
    with pytest.raises(SystemExit):
        main(['--root', str(root), '--output', str(root/'report.json')])


def test_invalid_clock_and_empty_root(tmp_path):
    assert build_report(tmp_path, now=100)['tasks'] == []
    with pytest.raises(ValueError):
        build_report(tmp_path, now=float('nan'))
    p = launch(tmp_path, start=9000)
    (p/'exit_code').write_text('not-an-int')
    t = build_report(tmp_path, now=8000)['tasks'][0]
    assert t['status'] == 'abnormal'
    assert t['resources']['allocated_seconds_upper_bound'] is None


def test_large_row_headers_skip_nested_status_and_cross_chunk_strings(tmp_path):
    path = tmp_path/'rollout_0000.json'; issues = []
    rows = [dict(status='complete', trajectory=[dict(status='error', text='x'*(4*1024*1024+5000))]),
            dict(trajectory=[dict(status='complete')], status='truncated'), {}, dict(status=None)]
    path.write_text(json.dumps(rows, indent=2)+'\n')
    counts, invalid, mode = training_statuses(path, issues, full_json_limit=0)
    assert counts == dict(complete=1, truncated=1) and invalid == 2
    assert mode == 'pretty_json_row_headers' and not issues
    path.write_text(json.dumps(rows, indent=2)[:-3])
    assert training_statuses(path, issues, full_json_limit=0) is None
    assert issues[-1]['code'] == 'unreadable_or_unsupported_result_headers'


def test_large_unknown_layout_is_flagged_instead_of_guessing_counts(tmp_path):
    path = tmp_path/'confirmation.json'; issues = []
    write(path, [dict(status='complete')])
    assert training_statuses(path, issues, full_json_limit=0) is None
    assert issues[0]['code'] == 'unreadable_or_unsupported_result_headers'


def test_exit_written_after_report_reference_time_is_capped(tmp_path):
    p = launch(tmp_path); exit_at(p, 0, 9000)
    t = build_report(tmp_path, now=8000)['tasks'][0]
    assert t['status'] == 'completed'
    assert t['resources']['endpoint'] == 'report_time_before_observed_exit'
    assert t['resources']['endpoint_unix'] == 8000


def test_optional_quick_view_never_guesses_skipped_game_counts(tmp_path, monkeypatch):
    launch(tmp_path)
    def forbidden(*_):
        raise AssertionError('quick view must not scan training trajectories')
    monkeypatch.setattr('boardbench.families.report.written_training', forbidden)
    report = build_report(tmp_path, now=8000, skip_training_records=True)
    assert not report['training_records_scanned']
    assert report['tasks'][0]['written_training'] is None


def test_complete_state_without_exit_is_not_process_completion_and_stale_exit_flags(tmp_path):
    p = launch(tmp_path)
    write(p/'state.json', dict(phase='complete', updated_unix=7000, next_update=10))
    t = build_report(tmp_path, now=8000)['tasks'][0]
    assert t['status'] == 'activity_unverified' and t['status_evidence'] == 'launch_without_exit'
    exit_at(p, 0, 3000)
    t = build_report(tmp_path, now=8000)['tasks'][0]
    assert t['status'] == 'abnormal'
    assert dict(code='exit_predates_launch', file='exit_code') in t['issues']
    exit_at(p, 0, 6000)
    t = build_report(tmp_path, now=8000)['tasks'][0]
    assert t['status'] == 'abnormal'
    assert dict(code='state_updated_after_exit_record', file='exit_code') in t['issues']
