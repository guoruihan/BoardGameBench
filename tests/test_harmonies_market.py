from copy import deepcopy
import random
import pytest
from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies import STACKS
from boardbench.environments.harmonies_market import HarmoniesMarket, FastHarmoniesMarket


def test_complete_fast_reference_and_supply():
    terminals=set()
    for seed in range(32):
        reference,fast=HarmoniesMarket(),FastHarmoniesMarket()
        obs,_=reference.reset(seed);assert fast.reset(seed)[0]==obs
        rng=random.Random(seed)
        while not obs['terminated']:
            assert reference.validate() and fast.validate()
            action=rng.choice(obs['legal_actions'])
            old_turn=obs['completed_turns'];old_discard=len(obs['discarded_cards'])
            obs=reference.step(action).observation
            assert fast.step(action).observation==obs
            if action['type']=='end_turn' and not obs['terminated']:
                assert obs['completed_turns']==old_turn+1
                assert len(obs['discarded_cards'])==old_discard+1
                assert sum(obs['discarded_tokens'].values())==3*obs['completed_turns']
        assert reference.validate() and fast.validate()
        assert obs['completed_turns']<=13
        terminals.add(obs['end_reason'])
    assert 'board_full_enough' in terminals


def test_slots_invalid_action_atomic_and_snapshot():
    game=HarmoniesMarket();game.reset(87)
    game.step(dict(type='choose_bundle',bundle_id=4));game.step(dict(type='take_card',market_slot=4))
    for action in (dict(type='choose_bundle',bundle_id=5),dict(type='end_turn',discard_card_slot=0)):
        state=game.save_state()
        with pytest.raises(InvalidAction): game.step(action)
        assert game.save_state()==state
    snap=game.save_state();other=FastHarmoniesMarket();assert other.load_state(snap)==game.observe()
    rng=random.Random(91)
    while not game.s['terminated']:
        a=rng.choice(game.legal_actions());assert game.step(a).observation==other.step(a).observation
    malformed=deepcopy(snap);malformed['state']['remaining_tokens']['blue']+=1
    state=game.save_state()
    with pytest.raises(ValueError): game.load_state(malformed)
    assert game.save_state()==state


def test_public_future_isolation_and_turn_cap():
    game=HarmoniesMarket();obs,_=game.reset(44)
    assert 'market_rng' not in obs and 'players' not in obs
    a,b=game.fork(1),game.fork(2)
    assert a.observe()==b.observe()==obs
    assert a._market_rng.getstate()!=b._market_rng.getstate()
    # Greedy stacking delays own-board saturation and witnesses the hard13 cap.
    obs,_=game.reset(0)
    while not obs['terminated']:
        actions=obs['legal_actions']
        placements=[a for a in actions if a['type']=='place_token']
        bundles=[a for a in actions if a['type']=='choose_bundle']
        if placements:
            action=max(placements,key=lambda a:len(STACKS[obs['board'][a['cell_id']]['stack_id']]))
        elif bundles:
            action=max(bundles,key=lambda a:sum(dict(gray=3,brown=3,red=2,green=1,blue=0,yellow=0)[c]
                       for c in obs['token_market'][a['bundle_id']]))
        else:
            action=next((a for a in actions if a['type']=='end_turn'),actions[0])
        obs=game.step(action).observation
    assert obs['end_reason']=='turn_limit' and obs['completed_turns']==13 and game.validate()
