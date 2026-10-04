"""Bounded end-to-end PPO throughput checks; never mutate a running campaign.

Every case starts with the same per-architecture frozen weights, same hyperparameters
and seeds. One warmup update is reported separately. Timed work includes complete
rollouts and actual PPO updates, not synthetic GPU-only forward throughput.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from boardbench.artifacts.store import digest, read_json, source_id, source_files, source_root, write_json
from .registry import IMPLEMENTED, make_splits, training_config
from .training import Trainer


def benchmark(out, seconds=15., workers=(0, 2, 4, 7), device='cpu',
              reference_root=None, experiments=IMPLEMENTED, repeats=1):
    if not math.isfinite(seconds) or seconds <= 0 or repeats < 1:
        raise ValueError('positive bounded benchmark duration/repeats required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for relative in source_files():
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root()/relative, target)
    manifest = make_splits()
    write_json(out/'splits.json', manifest)
    references = {}
    if reference_root:
        status = read_json(Path(reference_root)/'status.json')
        for experiment in experiments:
            entry = status['latest_checkpoints'][experiment+'_911']
            path = Path(entry['resume']).with_suffix('.weights.pt')
            metadata = read_json(Path(entry['resume']).with_suffix('.json'))
            if digest(path) != metadata['weights_sha256']:
                raise ValueError('reference weight hash mismatch')
            target = out/'references'/f'{experiment}.weights.pt'
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            references[experiment] = {'path':str(target), 'sha256':digest(target), 'from':str(path)}
    record = {'source_id':source_id(), 'torch':torch.__version__, 'device':device,
        'host':os.uname().nodename, 'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'started_unix':time.time(), 'seconds_per_case':seconds, 'repeats':repeats,
        'references':references, 'cases':[], 'method':'one warmup then complete rollout+PPO batches; no checkpoint IO in timed region',
        'comparison':'fixed initial weights per architecture; stochastic trajectories differ between collectors'}
    write_json(out/'benchmark.json', record)
    for repetition in range(repeats):
        for experiment in experiments:
            for variant, count in [(experiment, 0)] + [(experiment+'_T16', w) for w in workers]:
                if source_id() != record['source_id']:
                    raise RuntimeError('benchmark source changed; keep partial results and rerun frozen code')
                config = training_config(variant, 911, manifest, count)
                initialized = time.monotonic()
                trainer = Trainer(config, device)
                try:
                    if experiment in references:
                        weights = torch.load(references[experiment]['path'], map_location=device, weights_only=True)
                        trainer.solver.model.load_state_dict(weights['state_dict'], strict=True)
                    warmup = trainer.ppo_batch()
                    warmup_seconds = time.monotonic() - initialized
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    begin = time.monotonic()
                    initial_actions, initial_decisions = trainer.steps, trainer.decision_count
                    rows = []
                    while time.monotonic() - begin < seconds:
                        rows.append(trainer.ppo_batch())
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    wall = time.monotonic() - begin
                    error = max(r['initial_ratio_max_error'] for r in [warmup]+rows)
                    finite = all(math.isfinite(v) for r in [warmup]+rows for v in r.values() if isinstance(v,float))
                    if not finite or error > 1e-4:
                        raise RuntimeError('throughput benchmark correctness gate failed')
                    case = {'experiment':variant, 'workers':count, 'repetition':repetition,
                        'config':config, 'warmup_seconds':warmup_seconds, 'measured_seconds':wall,
                        'real_actions':trainer.steps-initial_actions, 'decisions':trainer.decision_count-initial_decisions,
                        'actions_per_second':(trainer.steps-initial_actions)/wall, 'batches':len(rows),
                        'initial_ratio_max_error':error, 'finite':finite,
                        'mean_inference_batch':sum(r.get('mean_inference_batch',1.) for r in rows)/len(rows),
                        'last_row':rows[-1]}
                    # Configuration is also written separately; do not repeat the large seed list in summary.
                    name=f'{variant}_workers{count}_repeat{repetition}'
                    write_json(out/f'{name}.config.json',case.pop('config'))
                    write_json(out/f'{name}.rows.json',rows)
                    record['cases'].append(case)
                    record['updated_unix']=time.time()
                    write_json(out/'benchmark.json',record)
                    print(json.dumps({k:v for k,v in case.items() if k!='last_row'}),flush=True)
                finally:
                    trainer.close()
                    del trainer
                    gc.collect()
                    if str(device).startswith('cuda'):
                        torch.cuda.empty_cache()
    record['ended_unix']=time.time()
    write_json(out/'benchmark.json',record)
    return record


def wait_for_idle(campaign, gpu, timeout):
    from boardbench.v12.launch import idle_gpu
    if gpu == 1:
        raise ValueError('GPU1 remains quarantined')
    deadline = time.monotonic() + timeout
    print(json.dumps({'event':'waiting_for_existing_campaign_evaluation_barrier',
                      'campaign':str(campaign),'gpu':gpu,'timeout':timeout}),flush=True)
    while time.monotonic() < deadline:
        status = read_json(Path(campaign)/'status.json')
        # No competing training dispatch while the old milestone's evaluations drain.
        if not status['active'] and not status['pending_jobs'] and time.time()-status['updated_unix'] < 30:
            try:
                uuid = idle_gpu(gpu)
            except RuntimeError:
                pass
            else:
                print(json.dumps({'event':'idle_gpu_acquired','gpu':gpu,'uuid':uuid}),flush=True)
                return uuid
        time.sleep(3.)
    raise TimeoutError('no non-disruptive GPU benchmark window; old campaign left untouched')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--workers',type=int,nargs='+',default=[0,2,4,7])
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--reference-root')
    parser.add_argument('--experiments',nargs='+',choices=IMPLEMENTED,default=list(IMPLEMENTED))
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--wait-campaign',help='wait for an existing campaign evaluation-only barrier')
    parser.add_argument('--gpu-index',type=int)
    parser.add_argument('--wait-seconds',type=float,default=5400.)
    args=parser.parse_args()
    if args.wait_campaign:
        if args.gpu_index is None or args.device != 'cuda':
            parser.error('wait-campaign requires gpu-index and device cuda')
        os.environ['CUDA_VISIBLE_DEVICES']=wait_for_idle(args.wait_campaign,args.gpu_index,args.wait_seconds)
    started = time.time()
    try:
        benchmark(args.out,args.seconds,args.workers,args.device,args.reference_root,args.experiments,args.repeats)
    finally:
        if Path(args.out).is_dir():
            write_json(Path(args.out)/'resource_interval.json',{
                'started_unix':started,'ended_unix':time.time(),
                'gpu_seconds':time.time()-started if args.device.startswith('cuda') else 0.,
                'cpu_reserved_cores':len(os.sched_getaffinity(0)),
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'accounting':'includes setup, warmup, all comparisons and teardown; excludes pre-acquisition wait'})
