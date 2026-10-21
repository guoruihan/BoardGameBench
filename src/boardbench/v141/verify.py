"""Paired actual-device parity and equal-resource throughput gate, no learning."""
import argparse
from copy import deepcopy
from pathlib import Path
import statistics
import time

import torch

from boardbench.artifacts.store import read_json, write_json, source_id
from boardbench.v140.training import Trainer
from boardbench.v140.encoding import ACTIONS, FEATURES


def parity(model, device):
    """Nonzero residual, actual dtype/device, every parameter's gradient."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        torch.manual_seed(141)
        candidate = deepcopy(model)
        torch.nn.init.normal_(candidate.context[-1].weight, std=.01)
        reference = deepcopy(candidate)
        x = torch.rand(17, FEATURES, dtype=torch.float64, device=device)
        mask = torch.rand(17, ACTIONS, device=device) > .35
        a, av = candidate(x, mask); b, bv = reference.forward_reference(x, mask)
        torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-11)
        torch.testing.assert_close(av,bv,atol=2e-11,rtol=2e-11)
        weight = torch.randn_like(a)
        ((a[mask]*weight[mask]).mean()+av.square().mean()).backward()
        ((b[mask]*weight[mask]).mean()+bv.square().mean()).backward()
        maximum = 0.
        for (name,p), (_,q) in zip(candidate.named_parameters(),reference.named_parameters()):
            if p.grad is None or q.grad is None:
                if p.grad is not None or q.grad is not None: raise RuntimeError('gradient routing: '+name)
            else:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-10,rtol=2e-9,msg=name)
                maximum=max(maximum,float((p.grad-q.grad).abs().max()))
        return dict(logits_max_error=float((a-b).abs().max()), value_max_error=float((av-bv).abs().max()),
                    gradient_max_error=maximum)


def run(config_path, checkpoint, out, workers=13, matches=128, repeats=3, minimum_speedup=1.2, device='cuda'):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    c=read_json(config_path); c.update(source_id=source_id(),workers=workers,batch_matches=matches)
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('actual CUDA required')
    t=Trainer(c,device)
    if checkpoint:
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        t.model.load_state_dict(state['model']);t.frozen.load_state_dict(state['frozen'])
        t.snapshots=[]
        for value in state['snapshots']:
            m=deepcopy(t.frozen);m.load_state_dict(value);t.snapshots.append(m)
    if not t.snapshots: t.snapshots=[deepcopy(t.frozen)]
    checks={}
    for objective in ('score','margin'):
        t.model.objective=objective;checks[objective]=parity(t.model,device)
    t.model.objective=c['objective']
    rows=[]
    seats=[i%2 for i in range(matches)]
    opponents=['frozen' if i%4<2 else 'lag0' if i%4==2 else 'current' for i in range(matches)]
    try:
        with t.make_pool(matches) as pool:
            # Complete-game warmup for both execution paths; same resources.
            for legacy in (True,False):
                t.collect(pool,list(range(190000000,190000000+matches)),seats,opponents,greedy=True,legacy=legacy)
            for repeat in range(repeats):
                results={};traces={}
                seeds=list(range(190010000+repeat*matches,190010000+(repeat+1)*matches))
                for legacy in ((True,False) if repeat%2==0 else (False,True)):
                    if device=='cuda':torch.cuda.synchronize()
                    tick=time.monotonic()
                    _,result=t.collect(pool,seeds,seats,opponents,greedy=True,legacy=legacy)
                    seconds=time.monotonic()-tick
                    name='legacy' if legacy else 'optimized'
                    results[name]=result
                    traces[name]=[[(d['seat'],d['action'],d['role']) for d in g['decisions']] for g in t.last_traces]
                    rows.append(dict(mode=name,repeat=repeat,seconds=seconds,
                                     **{k:v for k,v in result.items() if k!='per_game'}))
                    print({k:v for k,v in rows[-1].items() if k in ('mode','repeat','seconds','model_seconds','environment_seconds')},flush=True)
                if results['legacy']['per_game']!=results['optimized']['per_game'] or traces['legacy']!=traces['optimized']:
                    raise RuntimeError('paired greedy trajectories changed')
        times={name:statistics.median(r['seconds'] for r in rows if r['mode']==name) for name in ('legacy','optimized')}
        speedup=times['legacy']/times['optimized']
        result=dict(passed=speedup>=minimum_speedup,source_id=source_id(),parity=checks,
                    workers=workers,matches=matches,repeats=repeats,device=device,
                    gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                    checkpoint=str(checkpoint) if checkpoint else None, paired_greedy_trajectories_equal=True,
                    median_seconds=times,speedup=speedup,rows=rows,wall_seconds=time.monotonic()-start)
        write_json(out/'result.json',result)
        if not result['passed']:raise RuntimeError('no reliable end-to-end acceleration; do not replace training')
        print({k:v for k,v in result.items() if k!='rows'},flush=True)
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=13);p.add_argument('--matches',type=int,default=128)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--minimum-speedup',type=float,default=1.2)
    a=p.parse_args();run(a.config,a.checkpoint,a.out,a.workers,a.matches,a.repeats,a.minimum_speedup)
