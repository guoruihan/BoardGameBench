import json
import time
from pathlib import Path

import pytest

from boardbench.families import evaluate as ev

GAME = 'century_a_new_world'


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def simple_bots(tmp_path):
    return write(tmp_path/'bots.json', {'reference': {'kind': 'heuristic'}, 'random': {'kind': 'random'}})


def prepare(tmp_path, phase='dev', seeds=(51,)):
    bots = simple_bots(tmp_path)
    output = tmp_path/'evaluation'
    manifest = ev.prepare_evaluation(GAME, bots, output, list(seeds), phase)
    return output, manifest


def row(seed, orientation, a_points, status='complete', scores=(20, 10)):
    seats = ['a','b'] if orientation == 0 else ['b','a']
    rewards = [2*a_points-1, 1-2*a_points]
    if orientation:
        rewards.reverse()
    return dict(seed=seed, pair=['a','b'], orientation=orientation, seat_names=seats,
                match_id=f'{seed}_{orientation}', result=dict(status=status,
                    rewards=rewards if status == 'complete' else None, seconds=1.,
                    outcome=dict(scores=list(scores) if scores is not None else [None,None])))


def slow_worker(job):
    time.sleep(30)  # Owned subprocess must be killed by the evaluator deadline.
    raise AssertionError('deadline failed')


def test_same_seed_swap_fixed_first_player_and_complete_round_robin():
    schedule = ev.schedule_matches(GAME, ['c','a','b'], [91,92])
    assert len(schedule) == 12
    assert len({r['match_id'] for r in schedule}) == 12
    assert schedule == ev.schedule_matches(GAME, ['b','c','a'], [91,92])
    for seed in (91,92):
        for pair in (['a','b'],['a','c'],['b','c']):
            rows = [r for r in schedule if r['seed'] == seed and r['pair'] == pair]
            assert [r['seat_names'] for r in rows] == [pair,pair[::-1]]
            assert [r['first_player'] for r in rows] == [0,0]
    with pytest.raises(ValueError):ev.schedule_matches(GAME,['a','b'],[1,1])


def test_bootstrap_keeps_anticorrelated_seats_together():
    # Every environment is a paired draw. Independently bootstrapping its two
    # games would spuriously create a nonzero interval width.
    rows = [row(seed,orientation,1-orientation) for seed in range(12) for orientation in (0,1)]
    stats = ev.pair_statistics(rows,'a',bootstrap_samples=300)
    assert stats['completed_win_rate'] == .5
    assert stats['bootstrap']['completed_win_rate_ci95'] == [.5,.5]
    assert stats['bootstrap']['paired_complete_seed_win_rate_ci95'] == [.5,.5]
    assert stats['bootstrap']['independent_seeds'] == 12
    assert ev.pair_statistics(rows,'b',300)['completed_win_rate'] == .5


def test_bootstrap_seed_clusters_deterministic_and_wide_when_environments_differ():
    rows = [row(seed,orientation,float(seed%2)) for seed in range(8) for orientation in (0,1)]
    a = ev.pair_statistics(rows,'a',500,73)
    assert a == ev.pair_statistics(rows,'a',500,73)
    low, high = a['bootstrap']['completed_win_rate_ci95']
    assert low < .5 < high
    with pytest.raises(ValueError):ev.pair_statistics(rows[:-1],'a',10)


def test_all_incomplete_denominators_and_unknown_bounds_survive():
    rows = [row(0,0,1),row(0,1,0,'truncated'),row(1,0,0,'unresolved_rule'),
            row(1,1,0,'not_started'),row(2,0,0,'error'),row(2,1,0,'deadline')]
    stats = ev.pair_statistics(rows,'a',100)
    assert stats['scheduled'] == 6 and stats['completed'] == 1 and stats['incomplete'] == 5
    assert stats['completed_win_rate'] == 1
    assert stats['all_scheduled_win_rate'] == 1/6
    assert stats['all_scheduled_win_rate_bounds'] == [1/6,1]
    assert stats['paired_complete_seeds'] == 0
    assert stats['bootstrap']['paired_complete_seed_win_rate_ci95'] is None
    empty = ev.pair_statistics([row(8,0,0,'error'),row(8,1,0,'not_started')],'a',20)
    assert empty['completed_win_rate'] is None
    assert empty['all_scheduled_win_rate_bounds'] == [0,1]
    assert empty['bootstrap']['undefined_completed_resamples'] == 20


def test_score_means_follow_bot_seats_and_null_race_scores_stay_null():
    rows = [row(0,0,1,scores=(20,10)),row(0,1,1,scores=(30,40))]
    assert ev.pair_statistics(rows,'a',20)['score']['mean'] == 30
    assert ev.pair_statistics(rows,'b',20)['score']['mean'] == 20
    rows[0]['result']['outcome']['scores'] = [None,None]
    summary = ev.pair_statistics(rows,'a',20)
    assert summary['score'] == dict(mean=None,observed_count=1,missing_count=1)
    for r in rows:r['result']['outcome']['scores'] = [None,None]
    assert ev.pair_statistics(rows,'a',20)['score']['mean'] is None


def test_frozen_linear_spec_does_not_follow_best_json(tmp_path):
    weights = [1.]+[0.]*23
    artifact = write(tmp_path/'best.json',dict(kind='linear',weights=weights,generation=2))
    bots = write(tmp_path/'bots.json',{'learner':'best.json','reference':{'kind':'heuristic'}})
    output = tmp_path/'evaluation'
    manifest = ev.prepare_evaluation(GAME,bots,output,[17],'confirmation')
    original = manifest['bots']['learner']['identity_sha256']
    write(artifact,dict(kind='linear',weights=[9.]*24,generation=99))
    loaded = ev.load_manifest(output)
    assert loaded['bots']['learner']['spec']['weights'] == weights
    assert loaded['bots']['learner']['spec']['generation'] == 2
    assert loaded['bots']['learner']['identity_sha256'] == original
    assert loaded['phase'] == 'confirmation' and loaded['selection'].startswith('none')
    assert (output/'source.tar.gz').exists()


def test_model_bytes_are_copied_hashed_and_source_changes_do_not_affect_resume(tmp_path):
    original = tmp_path/'model.pt';original.write_bytes(b'frozen model bytes')
    artifact = write(tmp_path/'best.json',dict(kind='neural',path='model.pt'))
    bots = write(tmp_path/'bots.json',{'net':'best.json','heuristic':{'kind':'heuristic'}})
    output = tmp_path/'evaluation'
    manifest = ev.prepare_evaluation(GAME,bots,output,[17],'dev')
    bot = manifest['bots']['net']; frozen = Path(bot['spec']['path'])
    assert frozen != original and not frozen.is_symlink()
    assert bot['spec']['sha256'] == ev.file_hash(original)
    original.write_bytes(b'new checkpoint')
    write(artifact,dict(kind='neural',path='missing-new-model.pt'))
    assert ev.load_manifest(output)['bots']['net']['spec']['path'] == str(frozen)
    frozen.chmod(0o644);frozen.write_bytes(b'tampering')
    with pytest.raises(ValueError,match='model identity'):ev.load_manifest(output)


def test_bad_model_sha_and_missing_linear_weights_rejected(tmp_path):
    model = tmp_path/'model.pt';model.write_bytes(b'content')
    bots = write(tmp_path/'bots.json',{'a':dict(kind='neural',path='model.pt',sha256='bad'),
                                     'b':dict(kind='heuristic')})
    with pytest.raises(ValueError,match='checksum'):ev.freeze_bots(bots,tmp_path/'freeze',GAME)
    write(bots,{'a':dict(kind='linear'),'b':dict(kind='random')})
    with pytest.raises(ValueError,match='actual frozen weights'):ev.freeze_bots(bots,tmp_path/'freeze',GAME)


def test_manifest_and_source_identity_rejected_on_resume(tmp_path,monkeypatch):
    output,manifest = prepare(tmp_path)
    changed = ev.code_identity();changed['content_sha256'] = 'different'
    monkeypatch.setattr(ev,'code_identity',lambda:changed)
    with pytest.raises(ValueError,match='same code'):ev.load_manifest(output)
    manifest['seeds'] = [999]
    write(output/'manifest.json',manifest)
    with pytest.raises(ValueError,match='manifest identity'):ev.load_manifest(output,verify_code=False)


def test_confirmation_known_dev_seed_overlap_rejected(tmp_path):
    output,_ = prepare(tmp_path,seeds=(51,52))
    with pytest.raises(ValueError,match='overlap'):
        ev.prepare_evaluation(GAME,tmp_path/'bots.json',tmp_path/'confirm',[52],'confirmation',
                              exclude_manifests=[output/'manifest.json'])
    fresh = ev.prepare_evaluation(GAME,tmp_path/'bots.json',tmp_path/'confirm2',[53],'confirmation',
                                 exclude_manifests=[output/'manifest.json'])
    assert fresh['excluded_manifests'][0]['sha256'] == ev.file_hash(output/'manifest.json')


def test_expired_deadline_preserves_every_unstarted_job_then_resumes(tmp_path):
    output,manifest = prepare(tmp_path,phase='confirmation')
    stopped = ev.run_evaluation(output,workers=2,deadline_unix=time.time()-1,bootstrap_samples=50)
    assert stopped['scheduled'] == 2 and stopped['completed'] == 0
    assert stopped['statuses'] == {'not_started':2}
    resumed = ev.run_evaluation(output,workers=2,deadline_unix=time.time()+30,bootstrap_samples=50)
    assert resumed['scheduled'] == 2 and resumed['completed'] == 2
    assert resumed['selection_performed'] is False
    assert not (output/'best.json').exists()
    latest,attempts = ev.latest_attempts(output)
    assert len(attempts) == 2
    for row in ev.read_json(output/'games.json'):
        assert row['result']['seed'] == row['seed']
        assert [p['kind'] for p in row['result']['policies']] == [manifest['bots'][n]['spec']['kind'] for n in row['seat_names']]
    again = ev.run_evaluation(output,workers=1,deadline_unix=time.time()+10,bootstrap_samples=50)
    assert again['attempts'] == 2
    assert len(ev.latest_attempts(output)[1]) == 2


def test_running_deadline_is_preserved_as_attempt_and_resume_retries(tmp_path,monkeypatch):
    output,_ = prepare(tmp_path,seeds=(71,72))
    real = ev.frozen_match_worker
    monkeypatch.setattr(ev,'frozen_match_worker',slow_worker)
    began = time.monotonic()
    stopped = ev.run_evaluation(output,workers=1,deadline_unix=time.time()+.3,bootstrap_samples=20)
    assert time.monotonic()-began < 5
    assert stopped['statuses'] == {'deadline':1,'not_started':3}
    assert stopped['scheduled'] == 4 and stopped['attempts'] == 1
    monkeypatch.setattr(ev,'frozen_match_worker',real)
    resumed = ev.run_evaluation(output,workers=2,deadline_unix=time.time()+30,bootstrap_samples=20)
    assert resumed['completed'] == 4 and resumed['attempts'] == 5
    latest,attempts = ev.latest_attempts(output)
    assert sum(a['result']['status'] == 'deadline' for a in attempts) == 1
    assert sorted(a['attempt'] for a in attempts).count(2) == 1


def test_crash_running_attempt_retained_and_never_treated_as_loss(tmp_path):
    output,manifest = prepare(tmp_path)
    (output/'attempts').mkdir()
    job = manifest['schedule'][0]
    filename = job['match_id']+'_0001.json'
    write(output/'attempts'/filename,dict(match_id=job['match_id'],attempt=1,filename=filename,
        manifest_sha256=manifest['manifest_sha256'],
        started_unix=1.,result=dict(status='running',rewards=None,seconds=None)))
    stopped = ev.run_evaluation(output,deadline_unix=time.time()-1,bootstrap_samples=20)
    assert stopped['statuses'] == {'interrupted':1,'not_started':1}
    assert stopped['pair_reports'][0]['policies']['random']['all_scheduled_win_rate_bounds'] == [0,1]
    assert ev.latest_attempts(output)[1][0]['result']['status'] == 'interrupted'


def test_terminal_errors_are_preserved_and_not_retried_implicitly(tmp_path):
    output,manifest = prepare(tmp_path)
    (output/'attempts').mkdir()
    for job in manifest['schedule']:
        filename = job['match_id']+'_0001.json'
        write(output/'attempts'/filename,dict(match_id=job['match_id'],attempt=1,filename=filename,
        manifest_sha256=manifest['manifest_sha256'],
            result=dict(status='error',rewards=None,seconds=1.,error='fixture')))
    report = ev.run_evaluation(output,deadline_unix=time.time()+5,bootstrap_samples=20)
    assert report['statuses'] == {'error':2} and report['attempts'] == 2


def test_single_writer_lock_and_reference_cycles(tmp_path):
    with ev.evaluation_lock(tmp_path):
        with pytest.raises(RuntimeError,match='running writer'):
            with ev.evaluation_lock(tmp_path):pass
    artifact = write(tmp_path/'self.json',dict(spec_file='self.json'))
    bots = write(tmp_path/'bots.json',{'a':str(artifact),'b':dict(kind='random')})
    with pytest.raises(ValueError,match='Cyclic'):ev.freeze_bots(bots,tmp_path/'freeze',GAME)


def test_relative_paths_use_defining_json_directory_only(tmp_path,monkeypatch):
    package = tmp_path/'package'; package.mkdir()
    model = package/'weights.pt'; model.write_bytes(b'actual package model')
    artifact = write(package/'bot.json',dict(kind='neural',path='weights.pt'))
    bots = write(tmp_path/'bots.json',{'net':'package/bot.json','h':dict(kind='heuristic')})
    # A different same-named file in cwd must never influence package resolution.
    (tmp_path/'weights.pt').write_bytes(b'wrong model')
    monkeypatch.chdir(tmp_path)
    frozen = ev.freeze_bots(bots,tmp_path/'freeze',GAME)
    assert frozen['net']['model']['sha256'] == ev.file_hash(model)
    assert ev.file_hash(frozen['net']['spec']['path']) != ev.file_hash(tmp_path/'weights.pt')


def test_existing_attempts_cannot_be_reused_for_new_candidates(tmp_path):
    output = tmp_path/'evaluation';output.mkdir()
    (output/'attempts').mkdir()
    with pytest.raises(ValueError,match='empty directory'):
        ev.prepare_evaluation(GAME,simple_bots(tmp_path),output,[1],'dev')


def test_neural_frozen_worker_uses_common_loader_and_preserves_truncation(tmp_path):
    from boardbench.families.neural import CandidateActorCritic,save_model
    from boardbench.families.registry import game_class
    cls = game_class(GAME)
    model = tmp_path/'net.pt'
    save_model(model,CandidateActorCritic(hidden=4),cls,cls.default_weights)
    bots = write(tmp_path/'bots.json',{'net':dict(kind='neural',path='net.pt'),
                                     'h':dict(kind='heuristic')})
    frozen = ev.freeze_bots(bots,tmp_path/'freeze',GAME)
    result = ev.frozen_match_worker(dict(game_id=GAME,specs=[frozen['net']['spec'],frozen['h']['spec']],
                                         seed=1,first_player=0,max_decisions=1))
    assert result['status'] == 'truncated' and result['rewards'] is None
    assert result['worker_cpu_seconds'] >= 0


def test_frozen_model_package_can_move_as_a_whole(tmp_path):
    model = tmp_path/'model.pt';model.write_bytes(b'portable checkpoint')
    bots = write(tmp_path/'bots.json',{'net':dict(kind='neural',path='model.pt'),
                                     'h':dict(kind='heuristic')})
    output = tmp_path/'original'
    manifest = ev.prepare_evaluation(GAME,bots,output,[1],'dev')
    moved = tmp_path/'moved';output.rename(moved)
    loaded = ev.load_manifest(moved)
    assert loaded['bots']['net']['identity_sha256'] == manifest['bots']['net']['identity_sha256']
    assert Path(loaded['bots']['net']['spec']['path']).is_file()
    assert str(moved) in loaded['bots']['net']['spec']['path']


def test_harmonies_retained_policy_freezes_only_for_its_game(tmp_path):
    model = tmp_path/'retained.pt';model.write_bytes(b'identity-only fixture')
    bots = write(tmp_path/'bots.json',{'retained':dict(kind='harmonies_rl',path='retained.pt'),
                                     'random':dict(kind='random')})
    with pytest.raises(ValueError,match='only valid for Harmonies'):
        ev.freeze_bots(bots,tmp_path/'wrong',GAME)
    frozen = ev.freeze_bots(bots,tmp_path/'right','harmonies')
    assert frozen['retained']['spec']['kind'] == 'harmonies_rl'
    assert frozen['retained']['spec']['sha256'] == ev.file_hash(model)
