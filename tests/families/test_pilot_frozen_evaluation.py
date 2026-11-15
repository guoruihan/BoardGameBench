"""Evaluation-only reuse must preserve source bytes, deadlines and old cohorts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boardbench.families.pilot_run import (
    Checkpoints, PilotConfig, PilotRunner, blank_submission, default_queue, tree_hash,
)


def config(**changes):
    return replace(PilotConfig('frozen-rl','a'*64,'env-revision','codex-test',9999999999),**changes)


def frozen_source(tmp_path):
    source_root=tmp_path/'old-cohort'
    submission=source_root/'submission'; blank_submission(submission)
    (submission/'START_HERE.md').write_text('original frozen developed notes')
    checkpoints=Checkpoints(source_root/'checkpoints')
    checkpoints.checkpoint(submission)
    frozen=checkpoints.freeze()
    (source_root/'config.json').write_text(json.dumps(asdict(config())))
    (source_root/'results.jsonl').write_text('retained old costs and results\n')
    return frozen


class NoDevelopment:
    def __init__(self): self.matches=[]
    def run_development(self,*args): raise AssertionError('frozen cohort cannot develop')
    def run_match(self,context,match):
        assert context.checkpoints is None
        assert not (context.workspace/'prior_match').exists()
        assert not list(context.home.iterdir())
        text=(context.workspace/'submission/START_HERE.md').read_text()
        assert ('original frozen developed notes' in text) != match.control
        self.matches.append((match,tree_hash(context.workspace/'submission')))
        (context.workspace/'prior_match').write_text('private match feedback')
        (context.workspace/'submission/START_HERE.md').write_text('temporary game changes')
        return {'status':'natural_completion'}


def test_l20_frozen_cohort_has_20_independent_matches_and_preserves_source(tmp_path):
    source=frozen_source(tmp_path); original_hash=tree_hash(source)
    old_bytes={p.relative_to(source.parent.parent):p.read_bytes() for p in source.parent.parent.rglob('*') if p.is_file()}
    cfg=config(deployment_profiles=('L20',),protocol_version='pilot_v2_20261115_c4',
               calibration_seeds=(101,102,103,104,105))
    adapter=NoDevelopment(); root=tmp_path/'fresh-cohort'
    runner=PilotRunner(root,cfg,adapter)
    result=runner.run_frozen(source)
    assert result['status']=='complete' and result['development']=='reused_frozen'
    assert result['frozen_submission_sha256']==original_hash
    assert tree_hash(Path(result['frozen_submission']))==original_hash
    assert len(adapter.matches)==20
    developed=[m for m,h in adapter.matches if not m.control]
    control=[m for m,h in adapter.matches if m.control]
    assert len(developed)==len(control)==10
    assert {(m.seed,m.seat) for m in developed}=={(m.seed,m.seat) for m in control}
    assert all(m.profile=='L20' for m,h in adapter.matches)
    assert {h for m,h in adapter.matches if not m.control}=={original_hash}
    assert not (root/'phases/development').exists()
    assert json.loads((root/'config.json').read_text())['deployment_profiles']==['L20']
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    provenance=[event for event in events if event['event']=='frozen_submission_reused']
    assert len(provenance)==1 and provenance[0]['sha256']==original_hash
    assert provenance[0]['source_checkpoint']==str(source)
    assert not any(e['event'].startswith('development_') for e in events)
    assert old_bytes=={p.relative_to(source.parent.parent):p.read_bytes() for p in source.parent.parent.rglob('*') if p.is_file()}
    assert PilotRunner(root,cfg,adapter).run()['status']=='complete'
    assert len(adapter.matches)==20
    state_before=(root/'state.json').read_bytes()
    with pytest.raises(ValueError,match='fresh cohort'): runner.run_frozen(source)
    assert (root/'state.json').read_bytes()==state_before


@pytest.mark.parametrize('profiles',[(),('L20','L20'),('L0',),['L20'],(None,),(['L20'],)])
def test_profile_validation(profiles):
    with pytest.raises(ValueError,match='deployment_profiles'): config(deployment_profiles=profiles)


def test_default_queue_unchanged_and_control_always_l20():
    assert len(default_queue(config()))==40
    queue=default_queue(config(deployment_profiles=('L40',)))
    assert len(queue)==20
    assert {m.profile for m in queue if m.control}=={'L20'}
    assert {m.profile for m in queue if not m.control}=={'L40'}


def test_rejects_changed_source_and_extended_deadline(tmp_path):
    source=frozen_source(tmp_path)
    runner=PilotRunner(tmp_path/'new',config(deadline_unix=10000000000),NoDevelopment())
    with pytest.raises(ValueError,match='original deadline'): runner.run_frozen(source)
    assert not runner.state_path.exists()
    runner=PilotRunner(tmp_path/'other',config(),NoDevelopment())
    (source/'START_HERE.md').write_text('not the frozen artifact')
    with pytest.raises(ValueError,match='content changed'): runner.run_frozen(source)
    assert not runner.state_path.exists() and not list(runner.checkpoints.root.iterdir())


def test_existing_state_and_phase_budgets_never_reset(tmp_path):
    source=frozen_source(tmp_path); root=tmp_path/'existing'
    runner=PilotRunner(root,config(),NoDevelopment())
    ledger=root/'phases/old/ledger.json'; ledger.parent.mkdir(parents=True)
    ledger.write_text('{"spent": 123}')
    with pytest.raises(ValueError,match='fresh cohort'): runner.run_frozen(source)
    assert ledger.read_text()=='{"spent": 123}'
    assert not runner.state_path.exists()


def test_requires_frozen_record_and_rejects_symlink(tmp_path):
    submission=tmp_path/'loose'; blank_submission(submission)
    runner=PilotRunner(tmp_path/'new',config(),NoDevelopment())
    with pytest.raises(ValueError,match='trusted frozen'): runner.run_frozen(submission)
    source=frozen_source(tmp_path)
    (source/'link').symlink_to(source/'START_HERE.md')
    with pytest.raises(ValueError,match='links and special'): runner.run_frozen(source)
    assert not runner.state_path.exists()


def test_cli_exposes_options_without_starting_runtime():
    script=Path(__file__).resolve().parents[2]/'scripts/run_harmonies_pilot.py'
    result=subprocess.run([sys.executable,str(script),'--help'],text=True,capture_output=True,check=True)
    assert '--frozen-submission' in result.stdout and '--deployment-profiles' in result.stdout
    assert '--protocol-version' in result.stdout
