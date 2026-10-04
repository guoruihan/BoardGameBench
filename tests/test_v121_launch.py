import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from boardbench.artifacts.store import digest, read_json, write_json, source_files, source_id, source_root
from boardbench.v121 import launch
from boardbench.v121.registry import make_splits, training_config


def test_milestones_continue_and_overlapping_gpu_accounting():
    assert launch.next_milestone(4096)==400000
    assert launch.next_milestone(50000000)==100000000
    assert launch.next_milestone(100000000)==200000000
    assert launch.reservation_seconds([{'started':1.,'ended':11.},{'started':6.}],16.)==20.


def test_quarantined_gpu_rejected_before_any_launch(tmp_path):
    with pytest.raises(ValueError,match='quarantined'):
        launch.launch(tmp_path,tmp_path/'outputs/v121/run',[1])


def test_evaluation_pipeline_advances_training_without_dropping_backpressure():
    ready = launch.ready_for_next_target
    assert not ready([1], {}, [], None, {})
    assert not ready([], {0:1}, [], None, {})
    assert not ready([], {}, [1], None, {})  # Legacy barrier remains the default.
    assert not ready([], {}, [], {'running':True}, {})
    assert ready([], {}, [], None, {})
    allocation = {'pipeline_evaluations':True, 'evaluation_high_watermark':48}
    assert ready([], {}, [1]*20, {'running':True}, allocation)
    assert not ready([], {}, [1]*47, {'running':True}, allocation)
    assert not ready([], {}, [1]*48, None, allocation)


def test_accelerated_controller_trains_next_target_while_old_checkpoint_is_evaluated(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();(root/'logs').mkdir()
    for relative in source_files():
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root()/relative,target)
    cores=sorted(os.sched_getaffinity(0))[:1]
    write_json(root/'allocation.json',{'source_id':source_id(),'gpus':[{'index':99,'uuid':'cpu-test'}],
        'training_cpu_groups':[cores],'evaluation_cpu_cores':cores,'phase_seconds':30.,
        'training_seeds':[911],'experiments':['E00_T16'],'pipeline_evaluations':True})
    manifest=make_splits(train_count=8)
    config=training_config('E00_T16',911,manifest)
    config.update(batch_episodes=2,epochs=1,minibatch_size=64,purpose='interface_check')
    write_json(root/'plan/configs/E00_T16_911.json',config)
    write_json(root/'plan/splits.json',manifest)
    monkeypatch.setattr(launch,'MILESTONES',(1,1000000))
    monkeypatch.setattr(launch,'idle_gpu',lambda _: 'cpu-test')
    original=subprocess.Popen
    def spawn(command,**kwargs):
        command=list(command)
        if 'boardbench.v121.training' in command:
            command[command.index('--device')+1]='cpu'
        elif 'boardbench.v121.evaluation' in command:
            out=command[command.index('--out')+1]
            code=('import json,pathlib,sys,time; p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True);'
                  'r=pathlib.Path(sys.argv[2]); deadline=time.monotonic()+12; seen=False\n'
                  'while time.monotonic()<deadline:\n'
                  ' seen=any((r/"training").glob("E00_T16_911_a1000000*/training.jsonl"))\n'
                  ' if seen: break\n'
                  ' time.sleep(.05)\n'
                  '(p/"research_summary.json").write_text(json.dumps({"score_mean":0,"failures":0,"saw_next_training":seen}));'
                  '(r/"STOP_REQUESTED.json").write_text("{}")')
            command=[sys.executable,'-c',code,out,str(root)]
        return original(command,**kwargs)
    monkeypatch.setattr(launch.subprocess,'Popen',spawn)
    launch.controller(root)
    status=read_json(root/'status.json')
    assert status['phase']=='stopped_by_request' and status['error'] is None
    curve=read_json(root/'learning_curve.json')
    assert curve[0]['summary']['saw_next_training']
    assert curve[0]['target']==1 and status['current_target_actions']==1000000
    assert len(read_json(root/'gpu_intervals.json'))==2


@pytest.mark.parametrize('experiment',['E00','E00_T16','E00_T16_F64'])
def test_real_cpu_child_checkpoint_and_graceful_controller_stop(tmp_path,monkeypatch,experiment):
    root=tmp_path/'campaign';root.mkdir();(root/'logs').mkdir()
    for relative in source_files():
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root()/relative,target)
    cores=sorted(os.sched_getaffinity(0))[:1]
    write_json(root/'allocation.json',{'source_id':source_id(),'gpus':[{'index':99,'uuid':'cpu-test'}],
               'training_cpu_groups':[cores],'evaluation_cpu_cores':cores,'phase_seconds':30.,
               'training_seeds':[911],'experiments':[experiment]})
    manifest=make_splits(train_count=8)
    config=training_config(experiment,911,manifest)
    config.update(batch_episodes=2,epochs=1,minibatch_size=64,purpose='interface_check')
    write_json(root/'plan/configs'/f'{experiment}_911.json',config)
    write_json(root/'plan/splits.json',manifest)
    if experiment.endswith('_F64'):
        from boardbench.v121.training import Trainer
        from boardbench.v121.recovery import numeric_parent_config
        old = Trainer(numeric_parent_config(config)); old.ppo_batch()
        saved = tmp_path/'parent/old.pt'; old.save(saved); old.close()
        write_json(root/'recovery.json', {'entries':{experiment+'_911':{
            'resume':str(saved), 'actions':old.steps, 'numeric_import':True,
            'parent_source':source_id(), 'parent_sha256':digest(saved)}},
            'parent_target_actions':old.steps+1, 'prior_gpu_seconds':123., 'prior_cpu_core_seconds':984.})
        alloc=read_json(root/'allocation.json');alloc['recovery_sha256']=digest(root/'recovery.json')
        write_json(root/'allocation.json',alloc)
    monkeypatch.setattr(launch,'MILESTONES',(1,))
    monkeypatch.setattr(launch,'idle_gpu',lambda _: 'cpu-test')
    original=subprocess.Popen
    def spawn(command,**kwargs):
        command=list(command)
        if 'boardbench.v121.training' in command:
            command[command.index('--device')+1]='cpu'
        elif 'boardbench.v121.evaluation' in command:
            out=command[command.index('--out')+1]
            code=('import json,pathlib,sys; p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True);'
                  '(p/"research_summary.json").write_text(json.dumps({"score_mean":0,"failures":0}));'
                  'pathlib.Path(sys.argv[2]).write_text("{}")')
            command=[sys.executable,'-c',code,out,str(root/'STOP_REQUESTED.json')]
        return original(command,**kwargs)
    monkeypatch.setattr(launch.subprocess,'Popen',spawn)
    launch.controller(root)
    status=read_json(root/'status.json')
    assert status['phase']=='stopped_by_request' and status['error'] is None
    assert status['gpu_hours_cap'] is None and status['gpu_seconds']>0
    assert not status['final_test_executed'] and not status['deployment_changed']
    if experiment.endswith('_F64'):
        assert status['cumulative_gpu_seconds'] == status['gpu_seconds'] + 123.
        meta = read_json(Path(status['latest_checkpoints'][experiment+'_911']['resume']).with_suffix('.json'))
        assert meta['lineage']['parent_actions'] == old.steps
    assert status['experiment_status'][experiment]=='running'
    saved=Path(status['latest_checkpoints'][experiment+'_911']['resume'])
    assert saved.exists()
    assert launch.validate_child(saved.parent)['status']=='target_reached'
    records=read_json(root/'gpu_intervals.json')
    assert len(records)==1 and records[0]['returncode']==0 and 'ended' in records[0]
