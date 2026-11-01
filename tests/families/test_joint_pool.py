import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest
import torch

from boardbench.families.evaluate import canonical_hash, file_hash
from boardbench.families.joint_pool import (freeze_spec, absolute_spec, inside,
    load_fast, make_schedule, match, verify_pool, FORMAT)
from boardbench.families.joint_report import interval, pair_summary, strength_label, style_profiles
from boardbench.families.registry import game_class


def test_guided_dependency_survives_move_and_original_removal(tmp_path):
    from boardbench.families.neural import CandidateActorCritic, save_model as save_base
    from boardbench.families.guided_neural import GuidedActorCritic, GuidedPolicy, save_model
    from boardbench.families.rich_observation import PublicTreeEncoder
    torch.set_num_threads(1)
    source = tmp_path/'original'; source.mkdir(); cls = game_class('splendor')
    base = source/'base.pt'; save_base(base, CandidateActorCritic(16), cls, cls.default_weights)
    encoder = PublicTreeEncoder('splendor'); model = GuidedActorCritic(encoder.metadata['vocabulary_size'], 16)
    target = source/'guided.pt'
    save_model(target, model, encoder, dict(base_spec=dict(kind='neural', path=str(base), sha256=file_hash(base)), anchor_scale=.25))
    original = dict(kind='guided_rich', path=str(target), sha256=file_hash(target))
    env = cls(); obs = env.reset(174); legal = env.legal_actions()
    expected = GuidedPolicy('splendor', original).decide(obs, legal)
    root = tmp_path/'pool'; artifacts = {}; sources = {}
    frozen = freeze_spec(original, root, artifacts, sources)
    assert len(artifacts) == 2
    moved = tmp_path/'moved'; root.rename(moved); shutil.rmtree(source)
    actual = load_fast('splendor', absolute_spec(frozen, moved), 0).decide(obs, legal)
    assert actual == expected
    for name, sha in artifacts.items(): assert file_hash(moved/name) == sha
    with pytest.raises(ValueError): inside(moved, '../outside')
    with pytest.raises(ValueError): inside(moved, '/absolute')


def simple_manifest():
    cls = game_class('toy_battle'); members = {}
    for name in ('legacy', 'heuristic', 'progress'):
        spec = dict(kind='heuristic') if name != 'progress' else dict(kind='proposal', progress=True)
        identity = dict(game='toy_battle', config=cls.config_id, rules_version=str(cls.rules_version), spec=spec)
        members[name] = dict(spec=spec, policy_id=canonical_hash(identity))
    return dict(format=FORMAT, games={'toy_battle': dict(config=cls.config_id,
        rules_version=str(cls.rules_version), members=members)}, artifacts={})


def test_manifest_tampering_and_real_match(tmp_path):
    manifest = simple_manifest(); manifest['manifest_sha256'] = canonical_hash(manifest)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    assert verify_pool(tmp_path) == manifest
    job = dict(game='toy_battle', seed=1744, names=['legacy', 'heuristic'])
    result = match(tmp_path, job, manifest)
    assert result['status'] == 'complete'
    assert sum(result['seat_decisions']) > 0 and len(result['policy_ids']) == 2
    manifest['games']['toy_battle']['members']['legacy']['spec']['kind'] = 'random'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest identity'): verify_pool(tmp_path)


def test_single_match_cli_hash_and_no_overwrite(tmp_path, monkeypatch):
    from boardbench.families.joint_pool import main
    manifest = simple_manifest(); manifest['manifest_sha256'] = canonical_hash(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    job = dict(game='toy_battle', seed=1749, names=['legacy','heuristic'])
    (tmp_path/'job.json').write_text(json.dumps(job)); output = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', ['joint_pool', 'match', '--pool', str(tmp_path),
        '--job', str(tmp_path/'job.json'), '--output', str(output)])
    main(); result = json.loads(output.read_text()); sha = result.pop('result_sha256')
    assert result['status'] == 'complete' and canonical_hash(result) == sha
    with pytest.raises(FileExistsError): main()


def test_report_and_catalog_keep_policy_identity(tmp_path):
    from boardbench.families.joint_report import report
    manifest = simple_manifest()
    manifest['games']['toy_battle']['rules_coverage'] = {'status':'fixed_scope'}
    manifest['manifest_sha256'] = canonical_hash(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    schedule = make_schedule(manifest, fast_seeds=16, search_seeds=1)
    schedule['schedule_sha256'] = canonical_hash(schedule)
    (tmp_path/'schedule.json').write_text(json.dumps(schedule)); directory = tmp_path/'results'; directory.mkdir()
    for job in schedule['jobs']:
        names = job['names']; winner = names.index('legacy') if 'legacy' in names else 0
        result = dict(job=job, game=job['game'], seed=job['seed'], names=names,
            pool_sha256=manifest['manifest_sha256'], status='complete',
            policy_ids=[manifest['games']['toy_battle']['members'][n]['policy_id'] for n in names],
            outcome=dict(winners=[winner], scores=[4,5]), policy_seconds=[1.,1.],
            seat_decisions=[2,2], action_counts=[{'place':2},{'place':2}],
            score_breakdown=[{'medals':4},{'medals':5}])
        result['result_sha256'] = canonical_hash(result)
        (directory/(job['id']+'.json')).write_text(json.dumps(result))
    summary = report(tmp_path, tmp_path/'schedule.json', directory, tmp_path/'report')
    assert summary['recorded'] == summary['scheduled']
    assert summary['games']['toy_battle']['strength']['heuristic'] == 'below_anchor'
    catalog = json.loads((tmp_path/'report/opponent_catalog.json').read_text())
    assert catalog['games']['toy_battle']['roles']['anchor'] == 'legacy'
    assert catalog['games']['toy_battle']['members']['legacy']['policy_id'] == manifest['games']['toy_battle']['members']['legacy']['policy_id']


def test_schedule_has_swapped_seats_same_seeds_and_immutable_inputs():
    manifest = simple_manifest(); manifest['manifest_sha256'] = canonical_hash(manifest)
    schedule = make_schedule(manifest, fast_seeds=2, search_seeds=1)
    jobs = schedule['jobs']; ids = [j['id'] for j in jobs]
    assert len(ids) == len(set(ids))
    for i in range(0, len(jobs), 2):
        a, b = jobs[i:i+2]
        assert a['seed'] == b['seed'] and a['names'] == b['names'][::-1]
    assert min(j['seed'] for j in jobs) == 174010000
    with pytest.raises(ValueError): make_schedule(manifest, fast_seeds=0)
    with pytest.raises(ValueError): make_schedule(manifest, search_seeds=-2)
    with pytest.raises(ValueError): make_schedule(manifest, seed_start=-1)


def rows(n=16, status='complete'):
    jobs = []; results = []
    for seed in range(n):
        for swap in (0, 1):
            names = ['a','legacy'] if not swap else ['legacy','a']
            job = dict(seed=seed, swap=swap, names=names); jobs.append(job)
            seat = names.index('a')
            results.append(dict(job=job, seed=seed, names=names, status=status,
                outcome=dict(winners=[seat], scores=[10,20]), policy_seconds=[1,1]))
    return jobs, results


def test_incomplete_results_remain_unknown_and_cannot_promote():
    jobs, result = rows(); result[0]['status'] = 'truncated'
    summary = pair_summary(jobs, result, 'a')
    assert summary['complete'] == 31 and summary['complete_seed_pairs'] == 15
    assert summary['unknown_outcome_bounds'] == [31/32, 1.]
    assert strength_label(summary) == 'insufficient_evidence'


def test_extreme_small_samples_have_nonzero_uncertainty():
    assert interval([1.,1.,1.,1.])[0] < .6
    assert interval([0.,0.,0.,0.])[1] > .4
    jobs, result = rows()
    assert strength_label(pair_summary(jobs, result, 'a')) == 'above_anchor'
    assert strength_label(pair_summary(jobs, result, 'legacy')) == 'below_anchor'


def test_search_overrun_is_not_attributed_to_its_opponent():
    from boardbench.families.joint_report import budget_violations
    members = {'search':dict(spec=dict(kind='terminal_search', seconds=900.)),
               'legacy':dict(spec=dict(kind='heuristic'))}
    row = dict(job={'id':'example'}, names=['legacy','search'], search=[None, {'spent':907.9}])
    result = budget_violations([row], members)
    assert not result['legacy'] and result['search'][0]['spent'] == 907.9


def test_style_requires_common_opponent_and_both_seats():
    _, rs = rows()
    for r in rs:
        r.update(score_breakdown=[{'animals':10},{'animals':20}],
                 seat_decisions=[2,2], action_counts=[{'place':2},{'draw':2}])
    profiles = style_profiles(rs)
    assert profiles['a']['paired_seeds'] == 16 and 'legacy' not in profiles
    assert profiles['a']['means']['animals'] == 15
    assert not profiles['a']['descriptive_component_shifts']


def test_queue_deadline_terminates_only_own_children(tmp_path):
    from boardbench.families.joint_runner import run
    cpu = min(os.sched_getaffinity(0))
    plan = dict(output=str(tmp_path), cpus=[cpu], deadline_unix=time.time()+5.5,
                worker_seconds_limit=100, groups=[dict(id='slow', timeout=60,
                argv=[sys.executable,'-c','import time;time.sleep(60)'])])
    began = time.monotonic(); state = run(plan)
    assert time.monotonic()-began < 6 and not state['active'] and state['stopped']
