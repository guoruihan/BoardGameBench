import importlib.util
from pathlib import Path

from boardbench.environments.harmonies import Harmonies


def test_animal_browser_fixture_legal_complete_and_exact_restore():
    path=Path(__file__).resolve().parents[1]/'scripts/animal_browser_fixture.py'
    spec=importlib.util.spec_from_file_location('animal_browser_fixture',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data=module.fixture()
    env=Harmonies()
    env.load_state(data['snapshot'])
    restored=Harmonies()
    for i,action in enumerate(data['actions'],1):
        restored.load_state(env.save_state())
        assert env.step(action)==restored.step(action)
        if i==data['milestones']['slot_released']:
            assert len(env.s['active_cards'])==3
            assert env.s['completed_cards'][0]['placed_count']==5
    assert env.observe()==data['final_observation']
    assert sum(c['animal'] is not None for c in env.s['board'])==5
