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
