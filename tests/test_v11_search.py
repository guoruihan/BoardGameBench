from types import SimpleNamespace

import pytest
from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers.board import BoardSolver


@pytest.mark.parametrize('task', ('micro_tiles','take_it_easy','harmonies'))
@pytest.mark.parametrize('unit', ('atomic_action','turn'))
def test_search_budget_depth_units_and_public_future(task,unit,tmp_path):
    env=TASKS[task].factory()
    obs,_=env.reset(751)
    count=[0]
    def tick(): count[0]+=1
    ctx=SimpleNamespace(reference_simulator=SearchAccess(env.fork,tick))
    before=env.save_state()
    solver=BoardSolver(7,method='search',width=3,rollouts=2,depth=2,
                       depth_unit=unit,max_simulation_steps=96,leaf='potential')
    action=solver.decide(obs,ctx)
    assert action in obs['legal_actions'] and env.save_state()==before
    assert count[0]==solver.last_search['simulation_steps']<=96
    assert solver.last_search['depth_unit']==unit
    if task!='harmonies' or unit=='atomic_action':
        assert solver.last_search['horizon_completed_rollouts']==6
    solver.save(tmp_path)
    restored=BoardSolver(7)
    restored.load(tmp_path)
    assert restored.depth_unit==unit and restored.leaf=='potential'
    assert restored.decide(obs,ctx)==solver.decide(obs,ctx)


def test_turn_horizon_reports_budget_limited_rollouts():
    env=TASKS['harmonies'].factory()
    obs,_=env.reset(751)
    solver=BoardSolver(7,method='search',width=1,rollouts=1,depth=2,depth_unit='turn',max_simulation_steps=1)
    ctx=SimpleNamespace(reference_simulator=SearchAccess(env.fork,lambda:None))
    solver.decide(obs,ctx)
    assert solver.last_search['simulation_steps']==1
    assert solver.last_search['horizon_completed_rollouts']==0
