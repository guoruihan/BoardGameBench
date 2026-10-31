"""Internal transition path for already enumerated legal simulation actions.

The public environment.step() contract remains unchanged. Simulation callers
select an index into the legal list they just enumerated. No reward/observation
copy is constructed only to be immediately discarded. Exceptions invalidate
that simulation; the live environment never uses this path.
"""
import random
from .state_io import copy_data


def terminated(env):
    return env.engine.s['terminated'] if env.game_id=='harmonies' else env.state['terminated']


def step_legal(env,actions,index):
    if type(index) is not int or not 0<=index<len(actions):raise ValueError('Legal action index')
    if terminated(env):raise ValueError('Cannot simulate past terminal state')
    action=actions[index]
    if env.game_id=='harmonies':
        env.engine.step(action)
    else:
        env._step(action)
        env.state['decision_index']+=1


def clone(env):
    result=type(env).__new__(type(env))
    if env.game_id=='harmonies':
        result.engine=type(env.engine).__new__(type(env.engine))
        result.engine.load_state(env.engine.save_state())
    else:
        result.state=copy_data(env.state)
        result.rng=random.Random();result.rng.setstate(env.rng.getstate())
    return result
