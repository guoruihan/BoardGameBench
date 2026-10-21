from copy import deepcopy
import random
import pytest

from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies_duo import HarmoniesDuo, FastHarmoniesDuo


@pytest.mark.parametrize('seed', range(8))
def test_reference_fast_full_games(seed):
    a, b = HarmoniesDuo(), FastHarmoniesDuo()
    obs, _ = a.reset(seed, seed % 2); other, _ = b.reset(seed, seed % 2)
    rng = random.Random(seed)
    for _ in range(768):
        assert obs == other
        assert a.validate() and b.validate()
        if obs['terminated']:
            break
        action = rng.choice(obs['legal_actions'])
        left, right = a.step(action), b.step(action)
        assert left.reward == right.reward
        obs, other = left.observation, right.observation
    assert obs['terminated']
    assert obs['players'][0]['completed_turns'] == obs['players'][1]['completed_turns']
    assert obs['scores'] == [s['total'] for s in obs['score_breakdown']]


def finish_turn(env):
    while not env.s['players'][env.s['active_player']]['bundle_taken']:
        env.step(next(a for a in env.legal_actions() if a['type'] == 'choose_bundle'))
    while any(env.s['players'][env.s['active_player']]['pending_tokens'].values()):
        env.step(next(a for a in env.legal_actions() if a['type'] == 'place_token'))
    return env.step({'type': 'end_turn'})


def test_shared_refill_and_rejections():
    env = HarmoniesDuo(); obs, _ = env.reset(814)
    assert len(obs['token_market']) == len(obs['animal_market']) == 5
    assert sum(obs['remaining_tokens'].values()) == 105
    original = deepcopy(obs['token_market']); cards = deepcopy(obs['animal_market'])
    env.step({'type': 'take_card', 'market_slot': 4})
    env.step({'type': 'choose_bundle', 'bundle_id': 4})
    assert env.s['animal_market'][4] is None  # Refill is not immediate.
    assert env.s['token_market'][4] is None
    finish_turn(env)
    assert env.s['token_market'][:4] == original[:4]
    assert env.s['animal_market'][:4] == cards[:4]
    assert sum(env.s['remaining_tokens'].values()) == 102
    assert env.s['active_player'] == 1
    for action in ({'type': 'end_turn', 'discard_card_slot': 1},
                   {'type': 'choose_bundle', 'bundle_id': True},
                   {'type': 'take_card', 'market_slot': 5}):
        old = env.save_state()
        with pytest.raises(InvalidAction): env.step(action)
        assert env.save_state() == old


def test_final_round_and_tie():
    env = HarmoniesDuo()
    # Fixture isolates the final-round rule; conservation is separately tested.
    env.s['final_round'] = True
    finish_turn(env)
    assert not env.s['terminated'] and env.s['active_player'] == 1
    finish_turn(env)
    assert env.s['terminated']
    assert [p['completed_turns'] for p in env.s['players']] == [1, 1]
    with pytest.raises(InvalidAction): env.step({'type': 'end_turn'})


def test_pouch_triggers_only_when_refill_impossible():
    env = HarmoniesDuo()
    # Isolated replenishment fixture; full-game conservation is tested above.
    env._bag = ['gray']*3
    finish_turn(env)
    assert not env._bag and not env.s['final_round'] and not env.s['terminated']
    finish_turn(env)
    assert env.s['terminated'] and env.s['end_reason'] == 'empty_pouch'


def test_official_tie_break_by_placed_animals():
    env = HarmoniesDuo()
    env.s['active_player'] = 1
    env.s['players'][0]['completed_turns'] = 1
    env.s['players'][0]['turn_index'] = 2
    env.s['final_round'] = True
    # Independent terminal tie fixture: force equal score but unequal cubes.
    env.score_breakdown = lambda: [{'total': 42}, {'total': 42}]
    env.s['players'][0]['board'][0] = {'stack_id': 3, 'animal': 1}
    finish_turn(env)
    assert env.s['scores'] == [42, 42] and env.s['winners'] == [0]


def test_snapshot_public_future_and_copy():
    env = FastHarmoniesDuo(); obs, _ = env.reset(22)
    env.step({'type': 'take_card', 'market_slot': 4})
    state = env.save_state(); copy = FastHarmoniesDuo(); copy.load_state(state)
    assert copy.save_state() == state
    public = env.observe(); fork = FastHarmoniesDuo.from_observation(public, 913)
    assert fork.observe() == public and fork.validate()
    other = FastHarmoniesDuo.from_observation(public, 914)
    assert other.observe() == fork.observe() and other._bag != fork._bag
    public['players'][0]['board'][0]['stack_id'] = 14
    assert env.s['players'][0]['board'][0]['stack_id'] == 0
    malformed = deepcopy(state); malformed['private']['bag'].pop()
    with pytest.raises(ValueError): env.load_state(malformed)
    assert env.save_state() == state
