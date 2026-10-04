"""Opt-in only: caller must reserve one GPU and an isolated CPU allocation first."""
import os

import pytest
import torch

from boardbench.v121.registry import THROUGHPUT, make_splits, training_config
from boardbench.v121.training import Trainer

pytestmark = pytest.mark.skipif(os.environ.get('BOARDBENCH_CUDA_CHECK') != '1',
                                reason='requires explicitly reserved CUDA resource')


@pytest.mark.parametrize('name', list(THROUGHPUT))
def test_cuda_update_checkpoint_rng_and_optimizer_continuation(name, tmp_path):
    assert torch.cuda.is_available()
    cfg = training_config(name, 911, make_splits(train_count=32), collector_workers=2)
    cfg.update(batch_episodes=4, epochs=1, minibatch_size=64, purpose='cuda_interface_check')
    a = Trainer(cfg, 'cuda')
    try:
        row = a.ppo_batch()
        assert row['initial_ratio_max_error'] <= 1e-4
        a.save(tmp_path/'cuda.resume.pt')
        a.ppo_batch()
        expected = {k:v.detach().cpu().clone() for k,v in a.solver.model.state_dict().items()}
        counters = (a.steps,a.stop_decisions,a.decision_count,a.updates)
        expected_optimizer = {k:{f:v.detach().cpu().clone() for f,v in s.items()}
                              for k,s in a.optimizer.state_dict()['state'].items()}
    finally:
        a.close()
    b = Trainer(cfg, 'cuda')
    try:
        b.load(tmp_path/'cuda.resume.pt')
        row = b.ppo_batch()
        assert row['initial_ratio_max_error'] <= 1e-4
        assert (b.steps,b.stop_decisions,b.decision_count,b.updates) == counters
        for key,value in b.solver.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], atol=1e-6, rtol=1e-5)
        for key,state in b.optimizer.state_dict()['state'].items():
            for field,value in state.items():
                torch.testing.assert_close(value.cpu(), expected_optimizer[key][field], atol=1e-6, rtol=1e-5)
    finally:
        b.close()
