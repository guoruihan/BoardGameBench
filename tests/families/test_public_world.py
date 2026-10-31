import random
import pytest

from boardbench.families.registry import GAME_IDS,game_class
from boardbench.families.play import make_policy
from boardbench.families.public_world import sampled_world

GAMES=[g for g in GAME_IDS if g!='five_tribes']


@pytest.mark.parametrize('game',GAMES)
def test_all_sampled_worlds_preserve_visible_state_and_valid_future(game):
    env=game_class(game)();env.reset(172081,first_player=1)
    spec={'kind':'heuristic'}
    if game=='targi':spec['decision_guard']='targi_trader_acyclic_v1'
    if game=='century_a_new_world':spec['decision_guard']='century_new_world_public_repeat_v1'
    policies=[make_policy(type(env),spec,p) for p in (0,1)]
    for i in range(400):
        if env.outcome()['terminated'] or not env.legal_actions():break
        if i%17==0:
            before=env.snapshot()
            world=sampled_world(env,env.current_player,172099+i,verify=True)
            repeat=sampled_world(env,env.current_player,172099+i,verify=True)
            assert world.snapshot()==repeat.snapshot()
            assert env.snapshot()==before
            for _ in range(4):
                if world.outcome()['terminated'] or not world.legal_actions():break
                world.step(world.legal_actions()[0]);world.validate()
        env.step(policies[env.current_player].decide(env.observe(),env.legal_actions()))


@pytest.mark.parametrize('game',GAMES)
def test_live_hidden_permutations_and_rng_do_not_change_sampled_world(game):
    env=game_class(game)();env.reset(172052)
    changed=game_class(game)();changed.restore(env.snapshot())
    if game=='harmonies':
        changed.engine._bag.reverse();changed.engine._deck.reverse()
    else:
        s=changed.state;changed.rng=random.Random(99999)
        if game=='cascadia':
            n=len(s['deck']);pool=list(reversed(s['deck']+s['removed']));s['deck'],s['removed']=pool[:n],pool[n:]
        elif game=='calico':s['deck'].reverse()
        elif game=='century_spice_road':s['merchant_deck'].reverse();s['order_deck'].reverse()
        elif game=='jaipur':
            s['deck'].reverse()
            for x in s['bonus_tokens'].values():x.reverse()
        elif game=='splendor':
            for x in s['decks']:x.reverse()
            s['removed_nobles'].reverse()
        elif game=='gizmos':
            from boardbench.families.games.gizmos import CARDS
            leftovers=[x for x in s['removed'] if CARDS[x]['level']==3]
            n=len(s['decks'][2]);pool=list(reversed(s['decks'][2]+leftovers))
            s['decks'][2]=pool[:n];s['removed']=['g0-2','g0-3']+pool[n:]
            for x in s['decks'][:2]:x.reverse()
        elif game=='century_a_new_world':
            s['deck'][1:]=reversed(s['deck'][1:])
            for x in s['bonus_stacks']:x[1:]=reversed(x[1:])
            s['removed_bonuses'].reverse()
        elif game=='stone_age':
            s['card_deck'].reverse();s['removed_buildings'].reverse()
            for x in s['building_stacks']:x[:-1]=reversed(x[:-1])
        elif game=='targi':s['tribe_deck'].reverse();s['goods_deck'].reverse()
        elif game in ('dominion','el_dorado'):
            s['players'][0]['deck'].reverse()
            p=s['players'][1];n=len(p['hand']);pool=list(reversed(p['deck']+p['hand']))
            p['hand'],p['deck']=pool[:n],pool[n:]
        elif game=='toy_battle':
            for p in (0,1):
                nr=len(s['reserves'][p]);pool=list(reversed(s['reserves'][p]+s['removed'][p]))
                s['reserves'][p],s['removed'][p]=pool[:nr],pool[nr:]
    assert changed.observe()==env.observe()
    a=sampled_world(env,env.current_player,734,verify=True)
    b=sampled_world(changed,changed.current_player,734,verify=True)
    assert a.snapshot()==b.snapshot()


def test_jaipur_publicly_taken_goods_remain_in_opponent_hand():
    env=game_class('jaipur')();env.reset(17223)
    action=next(a for a in env.legal_actions() if a['type']=='take')
    env.step(action)
    from boardbench.families.games.jaipur import GOODS
    for seed in range(12):
        world=sampled_world(env,1,seed,verify=True)
        assert world.state['hands'][0][GOODS.index(action['good'])]>=1


def test_dominion_revealed_moat_is_retained_in_opponent_hidden_hand():
    import json
    from pathlib import Path
    from boardbench.families.public_world import dominion_zone_counts
    prefix=json.loads((Path(__file__).parents[1]/'fixtures/families/dominion_reveal_prefix.json').read_text())
    env=game_class('dominion')();env.reset(prefix['seed'],prefix['first_player'])
    for action in prefix['actions']:env.step(action)
    obs=env.observe();event=obs['history'][-1]
    assert event['event']=='reveal' and event['player']!=env.current_player
    assert dominion_zone_counts(obs)[event['player']]['known']['hand']['moat']>=1
    for seed in range(12):
        world=sampled_world(env,env.current_player,seed,verify=True)
        assert 'moat' in world.state['players'][event['player']]['hand']
