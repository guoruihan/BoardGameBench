"""Protocol controls for resource curves; no hidden-score strategy selection."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from boardbench.families.pilot_run import (
    Budget, Checkpoints, PilotConfig, PilotRunner, DevelopmentResult, default_queue, blank_submission,
)

spec=importlib.util.spec_from_file_location('resource_study',Path(__file__).resolve().parents[2]/'scripts/run_harmonies_resource_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def config(**kwargs):
    return PilotConfig('opponent','a'*64,'env','codex',9999999999,**kwargs)


class Adapter:
    def run_development(self,context,resume):
        self.budget=context.budget
        context.checkpoints.checkpoint(context.workspace/'submission')
        return DevelopmentResult('session',finished=True)
    def run_match(self,context,match):
        assert (context.workspace/'submission/START_HERE.md').exists()
        return {'status':'complete'}


def test_configured_budget_development_only_and_paired_queue(tmp_path):
    budget=study.bundles()['b025'];adapter=Adapter()
    cfg=config(development_only=True,development_budget=budget,calibration_pairs=16,
               calibration_seeds=tuple(range(16)),deployment_profiles=('L20',),include_controls=False)
    runner=PilotRunner(tmp_path,cfg,adapter)
    assert runner.run()['status']=='complete'
    assert adapter.budget==budget and not (tmp_path/'results.jsonl').exists()
    queue=default_queue(cfg)
    assert len(queue)==32 and all(not x.control for x in queue)
    assert {(m.seed,m.seat) for m in queue}=={(i,seat) for i in range(16) for seat in (0,1)}


def test_archived_explicit_new_experiment_does_not_change_old_freeze(tmp_path):
    source=tmp_path/'source';adapter=Adapter()
    original=PilotRunner(source,config(development_only=True),adapter);original.run()
    marker=(source/'checkpoints/frozen.json').read_bytes();sha=json.loads(marker)['sha256']
    target=PilotRunner(tmp_path/'new',replace(config(),deadline_unix=9999999998,calibration_pairs=2,include_controls=False,deployment_profiles=('L20',)),adapter)
    assert target.run_archived_snapshot(source,sha)['status']=='complete'
    assert len((tmp_path/'new/results.jsonl').read_text().splitlines())==4
    assert (source/'checkpoints/frozen.json').read_bytes()==marker
    with pytest.raises(ValueError,match='fresh root'):target.run_archived_snapshot(source,sha)


def test_archived_hash_mutation_rejected(tmp_path):
    source=tmp_path/'source';adapter=Adapter()
    original=PilotRunner(source,config(development_only=True),adapter);original.run()
    sha=json.loads((source/'checkpoints/frozen.json').read_text())['sha256']
    (source/'checkpoints'/sha/'START_HERE.md').write_text('changed')
    target=PilotRunner(tmp_path/'new',config(),adapter)
    with pytest.raises(ValueError,match='hash mismatch'):target.run_archived_snapshot(source,sha)


def test_history_deduplicates_only_manifest_format_without_using_scores(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'state.json').write_text('{"status":"complete"}')
    candidate=tmp_path/'candidate';blank_submission(candidate)
    store=Checkpoints(root/'checkpoints');a=store.checkpoint(candidate)
    manifest=json.loads((candidate/'manifest.json').read_text())
    (candidate/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    b=store.checkpoint(candidate)
    (candidate/'START_HERE.md').write_text('different instructions')
    c=store.checkpoint(candidate)
    d=root/'phases/development';d.mkdir(parents=True)
    (d/'ledger.0.stdout').write_text('\n'.join(json.dumps(dict(type='item.completed',item=dict(type='command_execution',exit_code=0,aggregated_output=str(dict(ok=True,sha256=sha))))) for sha in [a,b,c,b]))
    versions=study.historical_versions(root)
    assert len(versions)==2 and versions[0]['aliases']==[a,b] and versions[1]['aliases']==[c]


@pytest.mark.parametrize('value',[0,-1,True])
def test_invalid_pair_count(value):
    with pytest.raises(ValueError):config(calibration_pairs=value)


def test_paired_study_reuses_committed_keys_not_results(tmp_path):
    import hashlib
    keys={k:'ab'*32 for k in ['dev-r0','dev-r1','dev-r2','evaluation']}
    private=tmp_path/'private';private.mkdir()
    (private/'seed_keys.json').write_text(json.dumps(keys))
    from dataclasses import asdict
    plan={'seed_commitments':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in keys.items()},
          'developments':[{'level':level,'budget':asdict(budget)} for level,budget in study.bundles().items() for _ in range(3)]}
    (tmp_path/'preregistered_plan.json').write_text(json.dumps(plan))
    assert study.paired_seed_keys(tmp_path)==keys
    keys['evaluation']='cd'*32;(private/'seed_keys.json').write_text(json.dumps(keys))
    with pytest.raises(ValueError,match='commitments'):study.paired_seed_keys(tmp_path)


def test_circuit_breaker_uses_infrastructure_status_even_with_zero_exit(tmp_path):
    (tmp_path/'experiment').mkdir()
    p=tmp_path/'experiment/state.json'
    p.write_text('{"status":"infrastructure_failure"}')
    assert study.infrastructure_failed(tmp_path,0)
    p.write_text('{"status":"no_valid_submission"}')
    assert not study.infrastructure_failed(tmp_path,0)
    assert study.infrastructure_failed(tmp_path,1)


def test_continuation_preserves_completed_failed_and_original_deadline(tmp_path):
    from dataclasses import asdict
    plan={'model':'gpt-6-astra','evaluation_pairs':16,'deadline':12345}
    (tmp_path/'preregistered_plan.json').write_text(json.dumps(plan))
    (tmp_path/'finished.json').write_text('{"status":"paused_infrastructure"}')
    specs=[dict(label=f'{name}-r0',level=name,repeat=0,budget=asdict(budget)) for name,budget in study.bundles().items()]
    for label,status in [('b025-r0','complete'),('b050-r0','infrastructure_failure'),('b100-r0','complete')]:
        p=tmp_path/('dev-'+label)/'experiment';p.mkdir(parents=True)
        (p/'state.json').write_text(json.dumps(dict(status=status,frozen_submission='/old/checkpoints/'+'a'*64)))
    p=tmp_path/'eval-b025-r0/experiment';p.mkdir(parents=True)
    (p/'state.json').write_text('{"status":"complete"}')
    (p/'results.jsonl').write_text('immutable old evidence\n')
    before=(tmp_path/'preregistered_plan.json').read_bytes()
    parent,remaining,retained,evaluations,omitted=study.continuation_selection(tmp_path,specs,1,'gpt-6-astra')
    assert parent['deadline']==12345 and [x['label'] for x in remaining]==['b200-r0']
    assert retained[0]['label']=='b025-r0' and evaluations[0]['label']=='b100-r0'
    assert omitted==[dict(label='b050-r0',reason='previous_infrastructure_failure')]
    assert (tmp_path/'preregistered_plan.json').read_bytes()==before
    with pytest.raises(ValueError,match='model/pairs'):study.continuation_selection(tmp_path,specs,1,'gpt-5.6-sol')


def test_dns_recovery_is_low_frequency_and_deadline_bounded(tmp_path):
    import socket
    config=tmp_path/'private.json';config.write_text('{"base_url":"https://test.invalid/v1"}')
    clock=[100.];attempts=[]
    def sleep(seconds):clock[0]+=seconds
    def resolve(*args,**kwargs):
        attempts.append(clock[0])
        if len(attempts)<3:raise socket.gaierror(-2,'injected')
        return [('resolved',)]
    assert study.wait_for_network(config,1000,tmp_path/'events.jsonl',not_before=280,
                                  clock=lambda:clock[0],sleep=sleep,resolve=resolve)
    assert attempts==[280,340,460]
    assert 'test.invalid' not in (tmp_path/'events.jsonl').read_text()
    attempts.clear();clock[0]=0
    assert not study.wait_for_network(config,50,tmp_path/'expired.jsonl',clock=lambda:clock[0],sleep=sleep,resolve=resolve)
    assert clock[0]==50 and attempts==[0]


def test_recovered_diagnostic_is_not_a_terminal_network_error(tmp_path):
    p=tmp_path/'experiment/phases/development';p.mkdir(parents=True)
    log=p/'ledger.0.stdout'
    log.write_text(json.dumps({'type':'item.completed','item':{'aggregated_output':'upstream_dns_failure'}})+'\n')
    assert not study.known_preflight_network_failure(tmp_path)
    with log.open('a') as f:f.write(json.dumps({'type':'turn.failed','error':{'message':'502 upstream_dns_failure'}})+'\n')
    assert study.known_preflight_network_failure(tmp_path)


def test_reused_evaluation_uses_fixed_seed_prefix_without_replaying(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    source=tmp_path/'old';source.mkdir()
    records=[]
    for pair,score in [(0,10),(1,100)]:
        for seat in (0,1):
            records.append(dict(match_id=f'developed-L20-s{pair}-p{seat}',profile='L20',control=False,
                seed=pair,seat=seat,submission_sha256='a'*64,result=dict(status='complete',natural=True,
                gt=dict(outcomes=[dict(own_score=score,opponent_score=20,win_points=int(score>20))]))))
    (source/'results.jsonl').write_text('\n'.join(map(json.dumps,records))+'\n')
    retained=dict(label='b025-r0',source_experiment=str(source),results_sha256=hashlib.sha256((source/'results.jsonl').read_bytes()).hexdigest())
    plan=dict(retained_evaluations=[retained],historical_versions=[],developments=[],evaluation_pairs=1,development_repeats=1)
    rows=study.report(tmp_path,plan)
    assert len(rows)==1 and rows[0]['reused'] and rows[0]['recorded']==2
    assert rows[0]['own_score_mean']==10 and rows[0]['win_points_mean']==0
    (source/'results.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed'):study.report(tmp_path,plan)
