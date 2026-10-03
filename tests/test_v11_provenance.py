import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip('torch')
from boardbench.artifacts.store import read_json, write_json
from boardbench.benchmark import candidates, build_solver, validate, frozen_test
from boardbench.training import train


def small_run(path, *, hidden=128, seed=17, start=100, episodes=4,
              validation=(200,), test=(201,), task='micro_tiles'):
    train(dict(task_id=task, training_seed=seed, episodes=episodes, batch_episodes=1,
               epochs=1, threads=1, device='cpu', hidden=hidden,
               train_env_seed_start=start, validation_seeds=list(validation),
               test_seeds=list(test)), path)
    return path


def test_nondefault_widths_complete_pipeline_and_fresh_inference(tmp_path):
    primary = small_run(tmp_path/'primary', hidden=32)
    extra = small_run(tmp_path/'extra', hidden=64, seed=18, start=300)
    selection = validate('micro_tiles', primary, tmp_path/'validation', extra)
    definitions = selection['candidates']
    rl = [c for c in definitions if c['method']=='rl']
    assert len(rl) == 6 and len({c['id'] for c in rl}) == 6
    assert {c['hidden'] for c in rl} == {32, 64}
    assert len({c['origin']['run_id'] for c in rl}) == 2
    for c in rl:
        assert c['origin']['config']['hidden'] == c['hidden']
        assert c['origin']['splits']['train']
        assert c['origin']['rules_version'] == 'micro_tiles_v1'
        assert build_solver('micro_tiles', c).hidden == c['hidden']
    results = frozen_test(tmp_path/'validation/selection.json', tmp_path/'test', tmp_path/'policies')
    assert len(results) == 5 and all(r['completed']==1 for r in results.values())
    chosen = next(c for c in selection['selected'] if c['method']=='rl' and not c['untrained'])
    control = next(c for c in selection['selected'] if c.get('untrained'))
    assert control['origin']['run_id'] == chosen['origin']['run_id']
    checkpoint = tmp_path/'policies'/chosen['id']/'export/checkpoints/ep_000000'
    script = '''
import sys, torch
from boardbench.runner.engine import evaluate
torch.set_num_threads(1)
r=evaluate(sys.argv[1], {'episode_seeds':[777]}, sys.argv[2])
assert r['completed_episodes']==1 and r['episode_errors']==0, r
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(checkpoint), str(tmp_path/'new_process')],
                   cwd=tmp_path, check=True)
    manifest = read_json(tmp_path/'policies'/chosen['id']/'manifest.json')
    assert manifest['origin'] == chosen['origin']


def test_extra_cross_contamination_rejected_before_solver_build(tmp_path, monkeypatch):
    primary = small_run(tmp_path/'primary')
    extra = small_run(tmp_path/'extra', episodes=6, start=200, validation=(400,), test=(500,))
    def should_not_build(*args, **kwargs):
        pytest.fail('solver built before cross-run split preflight')
    monkeypatch.setattr('boardbench.benchmark.build_solver', should_not_build)
    with pytest.raises(ValueError, match='overlap'):
        validate('micro_tiles', primary, tmp_path/'validation', extra)
    assert not (tmp_path/'validation').exists()


def test_distinct_runs_same_checkpoint_names_retained_and_relocatable(tmp_path):
    primary = small_run(tmp_path/'primary')
    extra = small_run(tmp_path/'extra', seed=18, start=300)
    result = validate('micro_tiles', primary, tmp_path/'validation', extra)
    rl = [c for c in result['candidates'] if c['method']=='rl']
    assert len(rl)==6 and len({c['id'] for c in rl})==6
    relocated = tmp_path/'relocated'
    shutil.copytree(extra, relocated)
    old = [c['id'] for c in candidates(extra) if c['method']=='rl']
    new = [c['id'] for c in candidates(relocated) if c['method']=='rl']
    assert old == new


def test_primary_validation_test_overlap_and_wrong_task_rejected(tmp_path):
    primary = small_run(tmp_path/'primary')
    splits = read_json(primary/'splits.json')
    splits['test'] = splits['validation']
    write_json(primary/'splits.json', splits)
    with pytest.raises(ValueError, match='overlap'):
        validate('micro_tiles', primary, tmp_path/'validation')
    splits['test'] = [201]
    write_json(primary/'splits.json', splits)
    with pytest.raises(ValueError, match='task'):
        validate('take_it_easy', primary, tmp_path/'wrong_task')


def test_parallel_validation_matches_serial_scores_and_retains_runs(tmp_path):
    primary=small_run(tmp_path/'primary',validation=(200,202))
    extra=small_run(tmp_path/'extra',seed=18,start=300,validation=(200,202))
    serial=validate('micro_tiles',primary,tmp_path/'serial',extra,retain_runs=True)
    parallel=validate('micro_tiles',primary,tmp_path/'parallel',extra,retain_runs=True,workers=2)
    assert len(serial['selected'])==7
    assert {c['origin']['run_id'] for c in serial['selected'] if c['method']=='rl'}=={c['origin']['run_id'] for c in parallel['selected'] if c['method']=='rl'}
    assert {k:v['score_mean'] for k,v in serial['summaries'].items()}=={k:v['score_mean'] for k,v in parallel['summaries'].items()}


def test_sharded_frozen_runner_matches_serial_and_keeps_raw_logs(tmp_path):
    primary=small_run(tmp_path/'primary',test=(901,902))
    selection=validate('micro_tiles',primary,tmp_path/'validation')
    # Cover one search and one trained model to keep the regression small.
    selection['selected']=[c for c in selection['selected'] if c['method']=='search' or (c['method']=='rl' and not c['untrained'])]
    write_json(tmp_path/'frozen.json',selection)
    a=frozen_test(tmp_path/'frozen.json',tmp_path/'serial_test',tmp_path/'serial_policies')
    b=frozen_test(tmp_path/'frozen.json',tmp_path/'sharded_test',tmp_path/'sharded_policies',workers=2,shard_size=1)
    assert a.keys()==b.keys()
    for cid in a:
        assert a[cid]['score_mean']==b[cid]['score_mean']
        assert a[cid]['completed']==b[cid]['completed']==2
        assert len(b[cid]['raw_runs'])==2
        records=[]
        for relative in b[cid]['raw_runs']:
            records += [json.loads(line) for line in (tmp_path/'sharded_test'/relative/'episodes.jsonl').read_text().splitlines()]
        assert [r['seed'] for r in records]==[901,902]
        assert (tmp_path/'sharded_test'/cid/'aggregation.json').exists()


@pytest.mark.parametrize('key,value', [('rules_version','unknown_rules'),('hidden',64),('network','action_mlp')])
def test_architecture_and_rules_metadata_remain_strict(tmp_path,key,value):
    import torch
    from boardbench.artifacts.store import digest
    run=small_run(tmp_path/'primary',hidden=32)
    records=read_json(run/'checkpoints.json')
    path=run/records[0]['path']
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    artifact[key]=value
    torch.save(artifact,path)
    records[0]['sha256']=digest(path)
    write_json(run/'checkpoints.json',records)
    with pytest.raises(ValueError,match='mismatch'):
        candidates(run)
