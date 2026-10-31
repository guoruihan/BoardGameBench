"""Prepare, but never launch, an explicitly resumed paused pilot in a new directory.

Invoke only after the operator lifts the pause. Remaining budget comes from the
verified pause record; the preparation clock starts immediately. Parameters,
optimizer, RNG, opponent pool and original files are preserved.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from .play import write_json
from .pool_training import file_hash


def prepare_resume(pause_record,game,arm,output,source_commit,harmonies_kernel=None):
    matches=[j for j in json.loads(Path(pause_record).read_text())['jobs'] if j['game']==game and j['arm']==arm]
    if len(matches)!=1:raise ValueError('Pause record must identify exactly one verified run')
    record=matches[0];source=Path(record['output']);output=Path(output).resolve()
    if output.exists():raise ValueError('Recovery requires a new output directory')
    if record.get('exit_code') is None or not record.get('finite_weights'):
        raise ValueError('Source was not verified stopped with finite weights')
    remaining=float(record['remaining_seconds'])
    if not 0<remaining<=3600:raise ValueError('Invalid original remaining budget')
    cfg=json.loads((source/'config.json').read_text())
    payload=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    original_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if payload['config_hash']!=original_hash:raise ValueError('Source config/checkpoint mismatch')
    if file_hash(cfg['initial_path'])!=cfg['initial_sha256']:raise ValueError('Initial weights changed')
    for item in payload['pool']['members'].values():
        spec=item['spec']
        if spec.get('sha256') and file_hash(spec['path'])!=spec['sha256']:
            raise ValueError('Archived opponent changed')
    if harmonies_kernel is not None:
        if game!='harmonies':raise ValueError('Harmonies kernel requires Harmonies game')
        from .harmonies_kernel import NativeKernel
        NativeKernel(harmonies_kernel)  # Verify before creating or changing artifacts.
    now=time.time();deadline=now+remaining
    old_state=deepcopy(payload['state'])
    prior_elapsed=old_state.get('elapsed_before_resume',0.)+cfg['hours']*3600-remaining
    cfg.update(output=str(output),source_commit=source_commit,pipeline='compact_v2',
               hours=remaining/3600,deadline_unix=deadline,budget_origin_deadline=deadline,
               resume_provenance=dict(source_output=str(source),source_commit=payload['config']['source_commit'],
                    source_checkpoint_sha256=file_hash(source/'checkpoint.pt'),remaining_seconds=remaining,
                    pause_record=str(Path(pause_record).resolve()),prior_phase=old_state['phase'],
                    starting_learner_decisions=old_state.get('learner_decisions',0),
                    starting_updates=old_state.get('next_update',0),starting_costs=deepcopy(old_state.get('costs',{}))))
    if harmonies_kernel is not None:cfg['harmonies_kernel']=harmonies_kernel
    state=payload['state'];state.update(phase='prepared_resume',deadline=deadline,began_unix=now,
        elapsed_before_resume=max(0.,prior_elapsed),resume_source=str(source))
    state.pop('error',None);state.pop('traceback',None)
    output.mkdir(parents=True)
    if state.get('static_data_ref'):
        old=state['static_data_ref'];new=output/'static_training_state.json.gz'
        shutil.copyfile(old['path'],new)
        if file_hash(new)!=old['sha256']:raise ValueError('Static data changed')
        state['static_data_ref']=dict(old,path=str(new))
    payload.update(config=cfg,config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
    temporary=output/'checkpoint.part';torch.save(payload,temporary);temporary.replace(output/'checkpoint.pt')
    write_json(output/'config.json',cfg)
    write_json(output/'resume_provenance.json',dict(cfg['resume_provenance'],old_state_phase=old_state['phase']))
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pause-record',required=True,type=Path)
    p.add_argument('--game',required=True);p.add_argument('--arm',choices=['A','B'],required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--source-commit',required=True)
    p.add_argument('--harmonies-kernel',type=Path,help='Optional explicitly built artifact.json; verify and pin in config')
    a=p.parse_args();kernel=json.loads(a.harmonies_kernel.read_text()) if a.harmonies_kernel else None
    cfg=prepare_resume(a.pause_record,a.game,a.arm,a.output,a.source_commit,kernel)
    print(json.dumps(dict(config=str(a.output/'config.json'),remaining_seconds=cfg['hours']*3600,
                          launched=False)))


if __name__=='__main__':main()
