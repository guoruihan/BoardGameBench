"""Bounded paired pipeline benchmarks using saved, public training episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .neural import load_model
from .play import write_json
from .pool_evaluator import bounded_jobs
from .pool_training import rollout, compact_rollout, audit_behavior
from .ppo import update
from .registry import game_class
from .trajectory_io import (encode_json, json_default, compact_episode, write_compressed,
                             write_episode_batch, read_episode_batch)


def legacy_write(path, data):
    """Exact pre-optimization scalar-streaming serializer, retained for comparison."""
    with gzip.open(path,'wt',compresslevel=1) as handle:
        json.dump(data,handle,default=json_default,allow_nan=False,separators=(',',':'))


def numpy_rows(data):
    for episode in data['games']:
        for row in episode['records']+episode.get('excluded_records',[]):
            for key in ('x','prior'):row[key]=np.asarray(row[key],dtype=np.float32)
    return data


def serialization(archives, output, trials):
    result=[]
    for index, archive in enumerate(archives):
        data=numpy_rows(read_episode_batch(archive))
        encoded=encode_json(data)
        timings={name:[] for name in ('legacy','buffered','encode','compress','write')}
        for trial in range(trials):
            old=output/f'{index}_old.gz';new=output/f'{index}_new.gz'
            start=time.perf_counter();legacy_write(old,data);timings['legacy'].append(time.perf_counter()-start)
            start=time.perf_counter();blob=encode_json(data);t1=time.perf_counter()
            compressed=gzip.compress(blob,compresslevel=1,mtime=0);t2=time.perf_counter()
            temporary=new.with_suffix('.part');temporary.write_bytes(compressed);temporary.replace(new);t3=time.perf_counter()
            timings['buffered'].append(t3-start);timings['encode'].append(t1-start)
            timings['compress'].append(t2-t1);timings['write'].append(t3-t2)
            assert gzip.decompress(old.read_bytes())==gzip.decompress(new.read_bytes())==encoded
        packed=[compact_episode(episode) for episode in data['games']]
        bytes_before=len(pickle.dumps(data['games'],protocol=5))
        bytes_after=len(pickle.dumps(packed,protocol=5))
        medians={k:statistics.median(v) for k,v in timings.items()}
        row=dict(game=data['games'][0]['result']['game'],archive=str(archive),trials=trials,
                 seconds=medians,speedup=medians['legacy']/medians['buffered'],
                 uncompressed_bytes=len(encoded),pickle_before_bytes=bytes_before,pickle_after_bytes=bytes_after,
                 ipc_reduction=bytes_before/bytes_after,lossless_bytes_verified=True)
        result.append(row);print(json.dumps(row),flush=True)
    write_json(output/'serialization.json',result)
    return result


def legacy_dispatch(pool, jobs, workers):
    results=[]
    for index in range(0,len(jobs),workers):
        results.extend(pool.map(rollout,jobs[index:index+workers],chunksize=1))
    return results


def end_to_end(archive, output, workers, device, trials):
    data=read_episode_batch(archive)
    jobs=[]
    for episode in data['games']:
        r=episode['result'];seat=r['learner_seat']
        jobs.append(dict(game=r['game'],path=r['policies'][seat]['path'],self_play=r['self_play'],
             learner_seat=seat,seed=r['seed'],opponent=r['policies'][1-seat],opponent_id=r['opponent_id'],
             policy_version=r['policy_version'],max_seconds=120.,gae_lambda=.95))
    if any(e['result']['status']!='complete' for e in data['games']):
        raise ValueError('Use a completed representative batch for exact paired pipeline replay')
    cls=game_class(jobs[0]['game']);results=[]
    torch.set_num_threads(1)
    if device=='cuda':torch.ones(32,device=device).sum().item();torch.cuda.synchronize()
    warm_started=time.time()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        list(pool.map(rollout,jobs[:workers],chunksize=1))
        warmup=time.time()-warm_started
        for trial in range(trials):
            weights,semantics={},{}
            for mode in (('legacy','optimized') if trial%2==0 else ('optimized','legacy')):
                model,_=load_model(jobs[0]['path'],cls,device)
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
                start=time.perf_counter()
                played=(legacy_dispatch(pool,jobs,workers) if mode=='legacy' else
                        bounded_jobs(pool,compact_rollout,jobs,time.time()+300,workers))
                sampled=time.perf_counter()
                metadata={k:v for k,v in data.items() if k!='games'}
                saved=output/f'{cls.game_id}_{trial}_{mode}.json.gz'
                if mode=='legacy':legacy_write(saved,dict(metadata,games=played))
                else:
                    saved=output/f'{cls.game_id}_{trial}_{mode}.jsonl.gz'
                    write_episode_batch(saved,metadata,played)
                archived=time.perf_counter()
                rows=[r for episode in played for r in episode['records']]
                audit_behavior(model,rows,device)
                metric=update(model,optimizer,rows,device,np.random.default_rng(1701),4,128,.01,.01)
                if device=='cuda':torch.cuda.synchronize()
                updated=time.perf_counter()
                weights[mode]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                restored=read_episode_batch(saved)
                semantics[mode]=encode_json([dict(records=e['records'],excluded=e['excluded_records'],
                    outcome=e['result']['outcome'],scores=e['result']['score_breakdown'],rewards=e['result']['rewards'])
                    for e in restored['games']])
                row=dict(game=cls.game_id,mode=mode,trial=trial,workers=workers,device=device,
                    games=len(played),learner_decisions=len(rows),sampling_seconds=sampled-start,
                    saving_seconds=archived-sampled,update_seconds=updated-archived,total_seconds=updated-start,
                    file_bytes=saved.stat().st_size,warmup_seconds=warmup,loss=metric['loss'])
                results.append(row);print(json.dumps(row),flush=True)
            assert semantics['legacy']==semantics['optimized'], 'Trajectory/return mismatch'
            assert all(torch.equal(v,weights['optimized'][k]) for k,v in weights['legacy'].items()), 'PPO update changed'
    write_json(output/f'{cls.game_id}_end_to_end.json',dict(results=results,identical_trajectories=True,identical_update=True))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--end-to-end',action='store_true')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.trials<1 or args.workers<1:raise ValueError('Positive counts required')
    if args.end_to_end:
        for archive in args.archives:end_to_end(archive,args.output,args.workers,args.device,args.trials)
    else:serialization(args.archives,args.output,args.trials)


if __name__=='__main__':main()
