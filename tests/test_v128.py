from copy import deepcopy
import random
import os
from pathlib import Path

import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')

from boardbench.environments.harmonies import Harmonies,CARDS
from boardbench.v128.environment import FastHarmonies
from boardbench.v121.collector import EnvironmentPool
from boardbench.v128.collector import EnvironmentPool as FastPool
from boardbench.v12.encoding import encode
from boardbench.v125.policies import GoalContinuation
from boardbench.v127.training import Trainer
from boardbench.artifacts.store import read_json
from boardbench.artifacts.store import digest
from boardbench.v127.launch import build_jobs,validate_allocation,physical_gpu_seconds


@pytest.mark.parametrize('policy',['random','habitat'])
def test_stepwise_equivalence_and_isolation(policy):
    rng=random.Random(18);animal_count=0
    for seed in range(12 if policy=='random' else 6):
        a,b=Harmonies(),FastHarmonies();oa,ia=a.reset(seed);ob,ib=b.reset(seed)
        assert oa==ob and ia==ib
        heuristic=GoalContinuation()
        while not oa['terminated']:
            before=deepcopy(ob)
            action=rng.choice(oa['legal_actions']) if policy=='random' else heuristic.decide(oa)
            animal_count+=action['type']=='place_animal'
            ra,rb=a.step(action),b.step(action)
            assert ob==before  # old observations never alias the next state's containers
            assert ra==rb and a.save_state()==b.save_state()
            oa,ob=ra.observation,rb.observation
            assert encode(oa)==encode(ob)
        b.load_state(a.save_state());assert b.observe()==a.observe()
    if policy=='habitat':assert animal_count>10


def test_compiled_predicate_all_cards_and_mutation_isolation():
    rng=random.Random(44);a,b=Harmonies(),FastHarmonies()
    for _ in range(250):
        for cell in a.s['board']:
            cell.update(stack_id=rng.randrange(15),animal=None if rng.random()<.95 else 1)
        a.s['active_cards']=[{'card_id':c,'instance_id':c,'placed_count':0} for c in rng.sample(list(CARDS),4)]
        b.s=deepcopy(a.s)
        assert a.legal_actions()==b.legal_actions()
    ob=b.observe();ob['board'][0]['stack_id']=999;ob['pending_tokens']['blue']=99
    ob['active_cards'][0]['placed_count']=99
    assert b.s==a.s


@pytest.mark.parametrize('workers',[0,2])
def test_pool_arrays_same_for_entire_games(workers):
    rng=np.random.default_rng(127)
    with EnvironmentPool(4,workers) as a,FastPool(4,workers) as b:
        x=a.reset([15,32,115,208]);y=b.reset([15,32,115,208])
        while True:
            for k in ('features','masks','done','stats'):np.testing.assert_array_equal(x[k],y[k])
            if x['done'].all():break
            slots=np.flatnonzero(~x['done']);chosen=[int(rng.choice(np.flatnonzero(x['masks'][i,:240]))) for i in slots]
            x=a.step(slots,chosen);y=b.step(slots,chosen)


def research_config():
    root=Path(__file__).resolve().parents[1];spec=read_json(root/'configs/v127_a218_online_ppo.json')
    if not (root/spec['dataset']).exists():pytest.skip('local research artifacts unavailable')
    config=build_jobs(spec,root)[0][1]
    return {**config,'batch_episodes':4,'workers':0,'epochs':1,'critic_batches':1,'critic_epochs':2,'minibatch':128}


def test_ppo_reference_fast_identical_updates():
    config=research_config();a=Trainer(config)
    a.batch();row=a.batch();weights={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    b=Trainer({**config,'environment_backend':'fast_v128'})
    b.batch();other=b.batch()
    assert row['games']==other['games'] and row['actions']==other['actions']
    for k,v in b.model.state_dict().items():assert torch.equal(v,weights[k])
    assert other['initial_ratio_max_error']<1e-4;b.close()


def test_physical_gpu_interval_union_and_packing():
    jobs=[{'gpu':'a','started_unix':0,'elapsed_seconds':10},
          {'gpu':'a','started_unix':2,'elapsed_seconds':5},
          {'gpu':'a','started_unix':12,'elapsed_seconds':3},
          {'gpu':'b','started_unix':2,'elapsed_seconds':5}]
    assert physical_gpu_seconds(jobs,99)==18
    jobs.append({'gpu':'a','started_unix':14})
    assert physical_gpu_seconds(jobs,20)==23
    validate_allocation(list(range(128)),['gpu0','gpu1'],15,4)
    with pytest.raises(ValueError):validate_allocation(list(range(64)),['gpu0','gpu1'],15,4)


def test_explicit_continuation_preserves_next_update_and_rejects_algorithm_change(tmp_path):
    config=research_config();a=Trainer(config);a.batch();a.batch();p=tmp_path/'parent.pt';a.save(p)
    spec={'checkpoint':str(p),'sha256':digest(p),'source_id':a.source}
    row=a.batch();weights={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    child_config={**config,'environment_backend':'fast_v128','continuation':spec,'target_actions':1000000}
    b=Trainer(child_config);b.import_sampling_continuation(spec);other=b.batch()
    assert row['games']==other['games'] and row['ppo_actions']==other['ppo_actions']
    for k,v in b.model.state_dict().items():assert torch.equal(v,weights[k])
    b.close()
    bad=Trainer({**child_config,'learning_rate':.01})
    with pytest.raises(ValueError,match='algorithm'):bad.import_sampling_continuation(spec)
    bad.close()


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V128_CUDA')!='1',reason='authorized first-two-GPU continuation probe only')
def test_real_v127_cuda_optimizer_rng_continuation(tmp_path):
    assert torch.cuda.is_available()
    repo=Path(__file__).resolve().parents[1]
    parent=repo/'outputs/v127/ppo_a218_20261020_a/trials/soft_b/final.pt'
    config=read_json(parent.parent/'config.json');meta=read_json(parent.with_suffix('.json'))
    spec={'checkpoint':str(parent),'sha256':meta['sha256'],'source_id':meta['source_id']}
    base={**config,'continuation':spec,'target_actions':10000000,'workers':4}
    a=Trainer(base,'cuda');a.import_sampling_continuation(spec);row=a.batch()
    weights={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    b=Trainer({**base,'environment_backend':'fast_v128'},'cuda');b.import_sampling_continuation(spec);other=b.batch()
    assert row['games']==other['games'] and row['ppo_actions']==other['ppo_actions']
    for k,v in b.model.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=1e-10,atol=1e-10)
    assert other['initial_ratio_max_error']<1e-4
    p=tmp_path/'resume.pt';b.save(p);expected=b.batch();weights={k:v.clone() for k,v in b.model.state_dict().items()};b.close()
    c=Trainer({**base,'environment_backend':'fast_v128'},'cuda');c.load(p);actual=c.batch()
    assert expected['games']==actual['games']
    for k,v in c.model.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=1e-10,atol=1e-10)
    c.close()
