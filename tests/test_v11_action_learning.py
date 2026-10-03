import math
import subprocess
import sys

import pytest

torch=pytest.importorskip('torch')
from boardbench.environments import TASKS
from boardbench.environments.numerical import encode, ACTION_COUNTS, features
from boardbench.solvers.action_features import action_features, policy_features, WIDTHS
from boardbench.solvers.neural import NeuralSolver
from boardbench.training import train
from boardbench.imitation import train_imitation
from boardbench.benchmark import build_solver, candidates


@pytest.mark.parametrize('task', ('micro_tiles','take_it_easy'))
def test_action_network_public_features_shapes_gradients_and_restore(task,tmp_path):
    env=TASKS[task].factory()
    obs,_=env.reset(74)
    solver=NeuralSolver(19,task,hidden=32,network='action_mlp')
    while not obs['terminated']:
        rows=action_features(obs)
        assert len(rows)==ACTION_COUNTS[task] and all(len(r)==WIDTHS[task] for r in rows)
        assert all(math.isfinite(v) for r in rows for v in r)
        encoded=policy_features(obs,'action_mlp')
        assert len(encoded)==len(features(obs))+ACTION_COUNTS[task]*WIDTHS[task]
        assert policy_features(env.fork(83).observe(),'action_mlp')==encoded
        x=torch.tensor(encoded)[None]
        mask=torch.tensor(encode(obs)['action_mask'])[None]
        logits,value=solver.model(x,mask)
        assert logits.shape==(1,ACTION_COUNTS[task]) and value.shape==(1,)
        action=solver.decide(obs,None)
        assert action in obs['legal_actions']
        obs=env.step(action).observation
    result=train(dict(task_id=task,training_seed=19,episodes=4,batch_episodes=2,
                      network='action_mlp',hidden=32,epochs=1,threads=1),tmp_path/'ppo')
    assert result['parameter_changed']
    cp=candidates(tmp_path/'ppo')[-1]
    restored=build_solver(task,cp)
    assert restored.network=='action_mlp' and restored.hidden==32
    code='''
import sys
from boardbench.benchmark import build_solver, candidates
from boardbench.environments import TASKS
c=candidates(sys.argv[1])[-1]
s=build_solver(sys.argv[2],c)
e=TASKS[sys.argv[2]].factory(); o,_=e.reset(777)
while not o['terminated']: o=e.step(s.decide(o,None)).observation
assert o['score'] is not None
'''
    subprocess.run([sys.executable,'-I','-c',code,str(tmp_path/'ppo'),task],check=True,cwd=tmp_path)


def test_imitation_is_explicit_trained_and_has_no_inference_teacher(tmp_path,monkeypatch):
    result=train_imitation(dict(algorithm='imitation',task_id='take_it_easy',training_seed=51,
                                episodes=4,batch_episodes=2,network='action_mlp',hidden=32,
                                epochs=1,threads=1,validation_seeds=[555],test_seeds=[777]),tmp_path/'bc')
    assert result['parameter_changed']
    candidate=candidates(tmp_path/'bc')[-1]
    assert candidate['method']=='imitation' and not candidate['untrained']
    solver=build_solver('take_it_easy',candidate)
    monkeypatch.setattr('boardbench.solvers.board.ranked_actions',lambda *a: pytest.fail('inference teacher called'))
    env=TASKS['take_it_easy'].factory()
    obs,_=env.reset(987)
    while not obs['terminated']:
        obs=env.step(solver.decide(obs,None)).observation
    assert obs['score'] is not None
