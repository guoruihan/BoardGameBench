"""Bounded BC / candidate-conditional soft-Q distillation with resumable epochs."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.solvers.neural import PolicyNetwork


def objective(logits, y, q, qmask, q_weight):
    bc=F.cross_entropy(logits,y)
    valid=qmask.sum(-1)>1
    aux=logits.sum()*0
    if valid.any():
        conditional=logits[valid].masked_fill(~qmask[valid],-1e9).log_softmax(-1)
        aux=-(q[valid]*conditional).sum(-1).mean()
    return bc+q_weight*aux,bc,aux


def export_model(model,path):
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez(stream,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    temporary.replace(path)


def train(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_path=out/'config.json'
    if config_path.exists() and read_json(config_path)!=config:
        raise ValueError('resume config mismatch')
    write_json(config_path,config)
    if digest(config['dataset'])!=config['dataset_sha256']:
        raise ValueError('dataset hash mismatch')
    if (out/'summary.json').exists():
        result=read_json(out/'summary.json')
        if result['checkpoint_sha256']!=digest(out/'best.npz') or result['source_id']!=source_id():
            raise ValueError('completed model hash mismatch')
        return result
    torch.set_num_threads(config.get('threads',1))
    torch.manual_seed(config['seed'])
    device=config.get('device','cpu')
    if device=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        torch.ones(1,device='cuda').sum().item()
    with np.load(config['dataset'],allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in a.files}
    if arrays['x'].shape[1]!=1346 or arrays['mask'].shape[1]!=240:
        raise ValueError('dataset encoding mismatch')
    if not arrays['mask'][np.arange(len(arrays['y'])),arrays['y']].all():
        raise ValueError('masked teacher action')
    validation=arrays['validation'].astype(bool)
    if set(arrays['seed'][validation]) & set(arrays['seed'][~validation]):
        raise ValueError('environment split leakage')
    tensor={k:torch.as_tensor(arrays[k],device=device) for k in ('x','mask','y','q','qmask')}
    train_ids=torch.as_tensor(np.flatnonzero(~validation),device=device)
    val_ids=torch.as_tensor(np.flatnonzero(validation),device=device)
    if not len(train_ids) or not len(val_ids): raise ValueError('empty data partition')
    model=PolicyNetwork(1346,240,config['hidden']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0003),weight_decay=.0001)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    best=float('inf');best_epoch=epoch0=updates=0;prior_seconds=0.
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location=device,weights_only=True)
        if state['source_id']!=source_id() or state['config']!=config:
            raise ValueError('resume source/config mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'].cpu())
        if device=='cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        epoch0=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];updates=state['updates']
        prior_seconds=state['elapsed_seconds']
    begin=time.monotonic()
    last_metrics=read_json(out/'progress.json') if epoch0 and (out/'progress.json').exists() else {}
    epoch=epoch0
    with (out/'curve.jsonl').open('a') as log:
        for epoch in range(epoch0+1,config.get('epochs',120)+1):
            model.train();order=train_ids[torch.randperm(len(train_ids),device=device)]
            losses=[]
            for batch in order.split(config.get('minibatch',256)):
                logits,_=model(tensor['x'][batch],tensor['mask'][batch])
                loss,bc,aux=objective(logits,tensor['y'][batch],tensor['q'][batch],tensor['qmask'][batch],config['q_weight'])
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                nll=correct=count=0
                for batch in val_ids.split(512):
                    logits,_=model(tensor['x'][batch],tensor['mask'][batch])
                    nll+=float(F.cross_entropy(logits,tensor['y'][batch],reduction='sum'))
                    correct+=int((logits.argmax(-1)==tensor['y'][batch]).sum());count+=len(batch)
            metric=nll/count
            if not np.isfinite(metric): raise RuntimeError('nonfinite validation')
            elapsed=prior_seconds+time.monotonic()-begin
            if metric<best:
                best=metric;best_epoch=epoch;export_model(model,out/'best.npz')
                write_json(out/'best_metrics.json',{'epoch':epoch,'validation_nll':metric,
                           'validation_accuracy':correct/count,'checkpoint_sha256':digest(out/'best.npz')})
            last_metrics={'epoch':epoch,'updates':updates,'train_loss':float(np.mean(losses)),
                          'validation_nll':metric,'validation_accuracy':correct/count,
                          'best_epoch':best_epoch,'best_nll':best,'elapsed_seconds':elapsed}
            log.write(json.dumps(last_metrics)+'\n');log.flush()
            write_json(out/'progress.json',last_metrics)
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else torch.empty(0,dtype=torch.uint8),
                   'epoch':epoch,'best':best,'best_epoch':best_epoch,'updates':updates,
                   'elapsed_seconds':elapsed,'source_id':source_id(),'config':config}
            torch.save(state,out/'latest.partial');(out/'latest.partial').replace(out/'latest.pt')
            print(json.dumps(last_metrics),flush=True)
            if elapsed>=config.get('max_seconds',1800) or epoch-best_epoch>=25:
                break
    changed=any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items() if k.startswith('actor.'))
    if not changed: raise RuntimeError('actor parameters did not update')
    if read_json(out/'best_metrics.json')['checkpoint_sha256']!=digest(out/'best.npz'):
        raise ValueError('selected checkpoint/metrics mismatch')
    result={**last_metrics,'status':'completed','checkpoint':str(out/'best.npz'),
            'checkpoint_sha256':digest(out/'best.npz'),'device':device,'seed':config['seed'],
            'hidden':config['hidden'],'q_weight':config['q_weight'],'train_records':len(train_ids),
            'validation_records':len(val_ids),'parameter_changed':changed,'source_id':source_id(),
            'gpu_seconds':(prior_seconds+time.monotonic()-begin) if device=='cuda' else 0.,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),
            'selected_checkpoint_metrics':read_json(out/'best_metrics.json'),
            'checkpoint_selection':'minimum train-internal held-environment hard-label NLL; no dev score selection'}
    write_json(out/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(train(read_json(a.config),a.out)))
