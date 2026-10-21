from copy import deepcopy

import numpy as np
import pytest
import torch

from boardbench.v138.baseline import build_model
from boardbench.v140.model import DuoPolicy
from boardbench.v140.encoding import FEATURES, ACTIONS
from boardbench.v140.training import infer_actions, Trainer
from test_v140 import duo_config
from test_v129 import artifact
from boardbench.artifacts.store import read_json, write_json, digest
from boardbench.v141.training import continuation_config, migrate


@pytest.mark.parametrize('objective', ['score', 'margin'])
@pytest.mark.parametrize('method', ['control', 'advice', 'gated', 'sticky'])
def test_forward_gradients(objective, method):
    torch.set_num_threads(1)
    torch.manual_seed(141)
    model = DuoPolicy(build_model(16, method), objective)
    # A zero-initialized residual would conceal gradient-routing failures.
    torch.nn.init.normal_(model.context[-1].weight, std=.01)
    torch.nn.init.normal_(model.solo.adapter[-1].weight, std=.01)
    ref = deepcopy(model)
    x = torch.rand(7, FEATURES, dtype=torch.float64)
    mask = torch.rand(7, ACTIONS) > .4
    expected, ev = ref.forward_reference(x, mask)
    actual, av = model(x, mask)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(av, ev, atol=2e-12, rtol=2e-12)
    assert torch.equal(actual[~mask], expected[~mask])
    weight = torch.randn_like(actual)
    ((actual[mask]*weight[mask]).mean()+av.square().mean()).backward()
    ((expected[mask]*weight[mask]).mean()+ev.square().mean()).backward()
    for (name,p), (rn,r) in zip(model.named_parameters(), ref.named_parameters()):
        assert name == rn
        if p.grad is None or r.grad is None:
            assert p.grad is None and r.grad is None, name
        else:
            torch.testing.assert_close(p.grad, r.grad, atol=2e-11, rtol=2e-10, msg=name)


def test_body_advice_single_call_and_state_dict():
    torch.set_num_threads(1)
    model = DuoPolicy(build_model(16))
    counts = {'body': 0, 'advice': 0}
    def hook(name):
        def count(*_): counts[name] += 1
        return count
    handles = [model.solo.body.register_forward_hook(hook('body')),
               model.solo.advice.register_forward_hook(hook('advice'))]
    state = deepcopy(model.state_dict())
    with torch.inference_mode(): model(torch.zeros(8, FEATURES, dtype=torch.float64), torch.ones(8,ACTIONS,dtype=torch.bool))
    for h in handles: h.remove()
    assert counts == {'body': 1, 'advice': 1}
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    with pytest.raises(ValueError): model(torch.zeros(1,FEATURES),torch.zeros(1,ACTIONS,dtype=torch.bool))


def test_identity_groups_and_behavior_probabilities():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = DuoPolicy(build_model(16)); frozen = deepcopy(model)
    with torch.no_grad(): frozen.context[-1].bias.add_(torch.randn(ACTIONS+1,dtype=torch.float64))
    roles = ['learner', 'frozen', 'current', 'frozen', 'learner', 'current']
    models = dict(learner=model, current=model, frozen=frozen)
    x = torch.rand(6,FEATURES,dtype=torch.float64).numpy()
    mask = (torch.rand(6,ACTIONS)>.4).numpy()
    rng = torch.Generator().manual_seed(89)
    before = rng.get_state().clone()
    actual, calls = infer_actions(models, roles, x, mask, 'cpu', rng, greedy=True)
    expected, old_calls = infer_actions(models, roles, x, mask, 'cpu', rng, greedy=True, legacy=True)
    assert calls == 2 and old_calls == 3
    assert torch.equal(rng.get_state(),before)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
    sampled, _ = infer_actions(models, roles, x, mask, 'cpu', rng)
    for i, role in enumerate(roles):
        assert mask[i,int(sampled[i,0])]
        logits, value = models[role](torch.from_numpy(x[i:i+1]),torch.from_numpy(mask[i:i+1]))
        lp = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(int(sampled[i,0])))
        assert abs(lp.item()-sampled[i,1]) < 2e-12
        assert abs(value.item()-sampled[i,2]) < 2e-12


def test_complete_greedy_collection_parity(duo_config):
    t = Trainer(duo_config)
    t.snapshots = [deepcopy(t.frozen)]
    seeds = [100000011, 100000012, 100000013, 100000014]
    seats = [0,1,0,1]; opponents = ['frozen','current','lag0','current']
    with t.make_pool(4) as pool:
        a, am = t.collect(pool,seeds,seats,opponents,greedy=True,legacy=True)
        traces = deepcopy(t.last_traces)
        b, bm = t.collect(pool,seeds,seats,opponents,greedy=True)
    assert am['per_game'] == bm['per_game']
    for old,new in zip(traces,t.last_traces):
        assert [(r['seat'],r['action'],r['role']) for r in old['decisions']] == [(r['seat'],r['action'],r['role']) for r in new['decisions']]
    assert am['total_player_actions'] == bm['total_player_actions']
    for old,new in zip(a,b):
        np.testing.assert_array_equal(old[0],new[0]); np.testing.assert_array_equal(old[1],new[1])
        np.testing.assert_allclose(old[2:],new[2:],atol=2e-12,rtol=2e-12)
    t.close()


def test_migration_preserves_all_state_and_protected_config(duo_config, tmp_path):
    c={**duo_config,'source_id':None,'deadline_unix':9999999999.}
    parent=tmp_path/'parent';parent.mkdir()
    write_json(parent/'config.json',c)
    t=Trainer(c);t.batch();t.best_metric=12.3;t.elapsed_seconds=71.
    t.save(parent/'best.pt');t.batch();t.elapsed_seconds=99.;t.save(parent/'latest.pt');t.close()
    write_json(parent/'interrupted.json',{'actions':t.actions})
    config=continuation_config(parent,None)
    bad={**config,'learning_rate':.04}
    with pytest.raises(ValueError,match='protected'):migrate(parent,tmp_path/'bad',bad)
    out=tmp_path/'continued';migrate(parent,out,config)
    for name in ('best','latest'):
        old=torch.load(parent/(name+'.pt'),weights_only=False)
        new=torch.load(out/(name+'.pt'),weights_only=False)
        assert new['config']==config
        for key in old:
            if key=='config':continue
            def equal(a,b):
                if isinstance(a,torch.Tensor):assert torch.equal(a,b)
                elif isinstance(a,dict):
                    assert a.keys()==b.keys()
                    for k in a:equal(a[k],b[k])
                elif isinstance(a,(tuple,list)):
                    assert len(a)==len(b)
                    for v,w in zip(a,b):equal(v,w)
                else:assert a==b
            equal(old[key],new[key])
        assert read_json(out/(name+'.json'))['sha256']==digest(out/(name+'.pt'))
    restored=Trainer(config);restored.load(out/'latest.pt')
    assert restored.actions==t.actions and restored.best_metric==12.3
    assert restored.elapsed_seconds==99. and len(restored.snapshots)==2
    assert restored.batch()['actor_change']>0;restored.close()
    with pytest.raises(FileExistsError):migrate(parent,out,config)
