"""Reconstruct native-independent public plans and both raw/shaped own rewards."""
import numpy as np
from boardbench.environments.harmonies_duo import player_view
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import encode_batch
from boardbench.v143.replay import replay as raw_replay
from .credit import plan_potential


def replay(game):
    planners=[Planner(),Planner()];rows={seat:[] for seat in game['learner_seats']}
    for trace,step in zip(game['decisions'],raw_replay(game,'score'),strict=True):
        obs=step['observation'];seat=step['acting_player']
        vector=planners[seat].vector(player_view(obs))
        phi=float(plan_potential(encode_batch([obs],[vector]))[0]) if game['progress_scale'] else 0.
        if abs(phi-trace['plan_potential'])>1e-6:raise ValueError('public potential mismatch')
        if trace['trainable']:rows[seat].append(trace)
        elif 'training_reward' in trace or 'mc_training_return' in trace:raise ValueError('frozen decision entered learning')
        yield step
    for trajectory in rows.values():
        phi=np.array([r['plan_potential'] for r in trajectory])*game['progress_scale']/150
        if abs(phi[0])>1e-10:raise ValueError('initial potential nonzero')
        raw=np.array([r['reward_to_next_own_decision'] for r in trajectory])
        training=np.array([r['training_reward'] for r in trajectory])
        if not np.allclose(training,raw+np.diff(np.r_[phi,0.]),atol=1e-10,rtol=0):raise ValueError('potential telescoping mismatch')
        if not np.allclose(np.cumsum(training[::-1])[::-1],
                [r['mc_training_return'] for r in trajectory],atol=1e-10,rtol=0):raise ValueError('replay MC target mismatch')
