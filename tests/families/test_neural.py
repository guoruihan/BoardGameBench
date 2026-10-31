import numpy as np
import pytest
import torch

from boardbench.families.neural import CandidateActorCritic, collate, finish_returns, load_model, prepare, save_model
from boardbench.families.registry import game_class


def test_padding_masks_values_and_gradients():
    torch.manual_seed(4)
    model = CandidateActorCritic(16)
    a = dict(x=np.ones((2, 24), np.float32), prior=np.array([0., -2.], np.float32))
    b = dict(x=np.zeros((7, 24), np.float32), prior=np.zeros(7, np.float32))
    single, value = model(*collate([a], 'cpu'))
    logits, values = model(*collate([a, b], 'cpu'))
    torch.testing.assert_close(single[0], logits[0, :2])
    torch.testing.assert_close(value[0], values[0])
    assert torch.isneginf(logits[0, 2:]).all()
    loss = -torch.log_softmax(logits, -1)[0, 0] + (values - 1).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.actor.weight.grad.abs().sum() > 0
    assert model.critic[-1].weight.grad.abs().sum() > 0


def test_returns_do_not_bootstrap_other_seat():
    records = [{'value': .2}, {'value': .4}]
    finish_returns(records, 1., gamma=1., gae_lambda=1.)
    assert records[0]['return'] == pytest.approx(1.)
    assert records[1]['return'] == pytest.approx(1.)
    loser = finish_returns([{'value': -.1}], -1.)
    assert loser[0]['return'] == pytest.approx(-1.)


def test_export_validation_and_public_prior(tmp_path):
    cls = game_class('jaipur')
    model = CandidateActorCritic(16)
    path = tmp_path/'policy.pt'
    save_model(path, model, cls, cls.default_weights)
    loaded, saved = load_model(path, cls)
    assert saved['game'] == 'jaipur'
    with pytest.raises(ValueError):
        load_model(path, game_class('splendor'))
    x, prior = prepare([[100.]+[0.]*23, [2.]+[0.]*23], cls.default_weights)
    assert np.isfinite(x).all() and prior[0] > prior[1]
    bad = torch.zeros((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        loaded(torch.zeros((1, 2, 24)), torch.zeros((1, 2)), bad)
