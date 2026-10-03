"""Explicit heuristic distillation control; no teacher calls at policy inference."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, write_json, source_files, source_root, source_id
from boardbench.environments import TASKS
from boardbench.environments.numerical import ACTION_COUNTS, encode, action_id
from boardbench.provenance import validate_splits
from boardbench.solvers.action_features import policy_features
from boardbench.solvers.board import ranked_actions
from boardbench.solvers.neural import NeuralSolver
from boardbench.training import parameter_hash


def train_imitation(config, output):
    if config.get('algorithm') != 'imitation' or config['task_id'] not in ('micro_tiles','take_it_easy'):
        raise ValueError('imitation control requires explicit algorithm and supported task')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    for relative in source_files():
        target=out/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root()/relative,target)
    code_version=source_id(out/'source')
    task,seed,device=config['task_id'],config['training_seed'],config.get('device','cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    episodes,batch=config['episodes'],config.get('batch_episodes',32)
    if episodes<2*batch or episodes%batch:
        raise ValueError('episode budget must be a batch multiple and allow two checkpoints')
    start_seed=config.get('train_env_seed_start',4200000)
    splits={'train':list(range(start_seed,start_seed+episodes)),
            'validation':config['validation_seeds'],'test':config['test_seeds']}
    validate_splits(splits)
    write_json(out/'splits.json',splits)
    torch.set_num_threads(config.get('threads',2))
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    solver=NeuralSolver(seed,task,hidden=config.get('hidden',64),device=device,network=config.get('network','action_mlp'))
    optimizer=torch.optim.Adam(solver.model.parameters(),lr=config.get('learning_rate',.001))
    initial=parameter_hash(solver.model)
    start=time.monotonic()
    completed=steps=updates=0
    saved=[]
    def save(label):
        metadata=dict(algorithm='imitation',teacher='documented_heuristic',teacher_fraction=.8,
                      inference_teacher_calls=0,training_seed=seed,episodes=completed,
                      environment_steps=steps,gradient_steps=updates,wall_seconds=time.monotonic()-start,
                      parameter_sha256=parameter_hash(solver.model),initial_parameter_sha256=initial,
                      parameter_count=sum(p.numel() for p in solver.model.parameters()),
                      device=str(device),device_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
                      threads=config.get('threads',2),code_version=code_version,
                      torch_version=str(torch.__version__),numpy_version=np.__version__)
        solver.training=metadata
        path=out/(label+'.pt')
        solver.save_weights(path)
        saved.append({'path':path.name,'sha256':digest(path),**metadata})
        write_json(out/'checkpoints.json',saved)
    save('untrained')
    interval=config.get('checkpoint_every',max(batch,(episodes//2//batch)*batch))
    if type(interval) is not int or interval<batch or interval%batch:
        raise ValueError('checkpoint_every must be a positive batch multiple')
    with (out/'training.jsonl').open('x') as log:
        while completed<episodes and time.monotonic()-start<config.get('max_wall_seconds',900):
            xs,masks,targets,scores=[],[],[],[]
            for _ in range(batch):
                env=TASKS[task].factory()
                obs,_=env.reset(splits['train'][completed])
                while not obs['terminated']:
                    ranked=ranked_actions(obs)
                    values=np.full(ACTION_COUNTS[task],-1e9,dtype=np.float32)
                    for value,action in ranked:
                        values[action_id(task,action)]=value
                    values=(values-values.max())/config.get('teacher_temperature',1.)
                    probabilities=np.exp(values)
                    probabilities/=probabilities.sum()
                    xs.append(policy_features(obs,solver.network))
                    masks.append(encode(obs)['action_mask'])
                    targets.append(probabilities)
                    action=ranked[0][1] if rng.random()<.8 else ranked[int(rng.integers(len(ranked)))][1]
                    obs=env.step(action).observation
                    steps+=1
                scores.append(obs['score'])
                completed+=1
            x=torch.tensor(np.asarray(xs,dtype=np.float32),device=device)
            mask=torch.tensor(masks,device=device)
            targets=torch.tensor(np.stack(targets),device=device)
            losses=[]
            for _ in range(config.get('epochs',4)):
                order=rng.permutation(len(xs))
                for begin in range(0,len(xs),config.get('minibatch_size',256)):
                    ids=torch.as_tensor(order[begin:begin+config.get('minibatch_size',256)],device=device)
                    logits,_=solver.model(x[ids],mask[ids])
                    loss=-(targets[ids]*logits.log_softmax(-1)).sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite imitation loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(solver.model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    updates+=1
                    losses.append(loss.item())
            row=dict(episodes=completed,environment_steps=steps,gradient_steps=updates,
                     loss=float(np.mean(losses)),mean_teacher_mixture_score=float(np.mean(scores)),
                     wall_seconds=time.monotonic()-start)
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(row),flush=True)
            if completed%interval==0 or completed==episodes:
                save(f'trained_{completed:06d}')
        if saved[-1]['episodes']!=completed:
            save(f'trained_{completed:06d}')
    summary=dict(status='completed' if completed==episodes else 'wall_budget',checkpoints=saved,
                 parameter_changed=parameter_hash(solver.model)!=initial,
                 training_wall_seconds=time.monotonic()-start,test_used=False)
    write_json(out/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    train_imitation(read_json(args.config),args.out)
