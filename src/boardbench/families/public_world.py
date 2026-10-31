"""Observation-consistent sampled worlds for reference search.

The trusted bridge retains deterministic rule-execution bookkeeping and replaces
ALL private random payloads using public catalogs/observations and a fresh RNG.
It never copies the live RNG into a simulation. Beliefs are approximate: hidden
choices are not weighted by an opponent policy, and past private observations
not present in the observation contract are not reconstructed.
"""
from collections import Counter
import importlib
import random

from .registry import game_class
from .state_io import copy_data


def remaining(catalog,used,rng):
    counts=Counter(list(catalog));counts.subtract(used)
    if any(n<0 for n in counts.values()):raise ValueError('Public component conservation mismatch')
    out=sorted(counts.elements(),key=lambda x:(type(x).__name__,str(x)))
    rng.shuffle(out)
    return out


def take(pool,count):
    if not 0<=count<=len(pool):raise ValueError('Impossible public zone size')
    out=pool[:count];del pool[:count];return out


def dominion_zone_counts(obs):
    """Recover deck/discard COUNTS from public events, without card locations."""
    counts=[dict(deck=5,hand=5,discard=0,in_play=0,cellar=None,
                 known={k:Counter() for k in ('deck','hand','discard','in_play')}) for _ in (0,1)]
    def draw(c,n):
        for _ in range(n):
            if not c['deck']:
                c['deck'],c['discard']=c['discard'],0
                c['known']['deck'],c['known']['discard']=c['known']['discard'],Counter()
            if not c['deck']:break
            for card,amount in list(c['known']['deck'].items()):
                c['known']['hand'][card]+=max(0,1-(c['deck']-amount))
                c['known']['deck'][card]=max(0,amount-1)
            c['deck']-=1;c['hand']+=1
    for event in obs['history']:
        p=event['player'];c=counts[p];typ=event['event'];card=event.get('card')
        if c['cellar'] is not None and typ!='discard':
            draw(c,c['cellar']);c['cellar']=None
        if typ=='play':
            c['hand']-=1;c['in_play']+=1
            c['known']['hand'][card]=max(0,c['known']['hand'][card]-1)
            c['known']['in_play'][card]+=1
            if card=='cellar':c['cellar']=0
            elif card in ('village','market','merchant','smithy','moat'):
                draw(c,{'smithy':3,'moat':2}.get(card,1))
        elif typ=='discard':
            c['hand']-=1;c['discard']+=1
            c['known']['hand'][card]=max(0,c['known']['hand'][card]-1)
            c['known']['discard'][card]+=1
            if c['cellar'] is not None:c['cellar']+=1
        elif typ=='gain':
            zone=event.get('destination','discard');c[zone]+=1;c['known'][zone][card]+=1
        elif typ=='trash':
            c['hand']-=1;c['known']['hand'][card]=max(0,c['known']['hand'][card]-1)
        elif typ=='reveal_moat':c['known']['hand']['moat']=max(1,c['known']['hand']['moat'])
        elif typ=='cleanup_top':
            c['known']['discard'].update(c['known']['hand']);c['known']['discard'].update(c['known']['in_play'])
            c['known']['discard'][card]=max(1,c['known']['discard'][card])
            c['known']['hand']=Counter();c['known']['in_play']=Counter()
            c['discard']+=c['hand']+c['in_play'];c['hand']=c['in_play']=0;draw(c,5)
    for p,c in enumerate(counts):
        if c['cellar'] is not None and not(obs['phase']=='cellar' and obs['active_player']==p):
            draw(c,c['cellar']);c['cellar']=None
        view=obs['players'][p]
        if c['hand']!=view['hand_count'] or c['in_play']!=len(view['in_play']):
            raise ValueError('Public Dominion event counts do not reconstruct this state')
        if p==obs['player'] and c['deck']!=view['draw_count']:
            raise ValueError('Public Dominion deck count mismatch')
    return counts


def _resample(s,o,game,rng):
    m=importlib.import_module(game_class(game).__module__)
    seat=o['player'];other=1-seat
    if game in ('patchwork','machi_koro','clever'):
        return # No hidden component zones; all future chance uses the new RNG.
    if game=='cascadia':
        used=[c['tile'] for b in o['boards'] for c in b]+[c for c in o['market_tiles'] if c is not None]
        if o['pending_tile'] is not None:used.append(o['pending_tile'])
        used=[c for c in used if c in m.DATA['normal_tile_ids']]
        pool=remaining(m.DATA['normal_tile_ids'],used,rng)
        s['deck']=take(pool,o['deck_size']);s['removed']=pool
        animals=[c['animal'] for b in o['boards'] for c in b if c['animal'] is not None]
        animals += [a for a in o['market_animals'] if a is not None]
        if o['pending_animal'] is not None:animals.append(o['pending_animal'])
        s['bag']={a:20-animals.count(a) for a in m.ANIMALS}
    elif game=='calico':
        available={face:[face*3+i for i in range(3)] for face in range(36)}
        def physical(face):
            if not available[face]:raise ValueError('Calico public tile multiplicity')
            return available[face].pop(0)
        s['boards']=[[None if face<0 else physical(face) for face in b[:22]] for b in o['boards']]
        s['hands'][seat]=[physical(c) for c in o['hand']]
        s['market']=[physical(c) for c in o['market']]
        # Drafts are public; retain a conservative known-hand lower bound.
        known=Counter()
        for e in o['public_history']:
            if e['player']!=other:continue
            if e['type']=='draft':known[e['tile']]+=1
            elif e['type']=='place' and known[e['tile']]:known[e['tile']]-=1
        hand=[physical(c) for c in sorted(known.elements())]
        pool=[c for ids in available.values() for c in ids];rng.shuffle(pool)
        s['hands'][other]=hand+take(pool,o['hand_sizes'][other]-len(hand));s['deck']=pool
    elif game=='century_spice_road':
        owned=[[f'start-{p}-produce',f'start-{p}-upgrade'] for p in (0,1)]
        claimed=[[],[]]
        for event in o['history']:
            a=event['action'];p=event['player']
            if a['type']=='acquire':owned[p].append(a['card'])
            if a['type']=='claim':claimed[p].append(a['card'])
        s['hands'][seat]=list(o['hand'])
        s['hands'][other]=[c for c in owned[other] if c not in o['played'][other]]
        s['claimed']=claimed
        used=[c for cs in owned for c in cs if not c.startswith('start-')]
        used += [e['card'] for e in o['merchant_market']]
        s['merchant_deck']=remaining([c['id'] for c in m.DATA['merchants']],used,rng)
        s['order_deck']=remaining(m.ORDERS,[c for cs in claimed for c in cs]+o['order_market'],rng)
    elif game=='jaipur':
        used=list(o['market'])+list(o['discard'])
        used += [g for g,n in zip(m.GOODS,o['hand']) for _ in range(n)]+['camel']*o['herd']
        pool=remaining([g for g,n in m.DATA['cards'].items() for _ in range(n)],used,rng)
        # The number of non-hand cards not in the visible deck determines the
        # opponent herd; it is not read from the private state.
        herd=len(pool)-o['deck_count']-o['opponent']['hand_size']
        if herd<0 or pool.count('camel')<herd:raise ValueError('Jaipur herd constraints')
        for _ in range(herd):pool.remove('camel')
        known=Counter()
        for event in o['history']:
            if event.get('type')=='round_start':known.clear();continue
            if event.get('player')!=other:continue
            a=event['action']
            if a['type']=='take':known[a['good']]+=1
            elif a['type']=='sell':known[a['good']]=max(0,known[a['good']]-a['count'])
            elif a['type']=='exchange':
                for g,n,t in zip(m.GOODS,a['give'],a['take']):known[g]=max(0,known[g]-n)+t
        goods=remaining([g for g in pool if g!='camel'],known.elements(),rng)
        hand=list(known.elements())+take(goods,o['opponent']['hand_size']-sum(known.values()))
        for g in hand:pool.remove(g)
        rng.shuffle(pool)
        s['herds'][other]=herd;s['hands'][other]=[hand.count(g) for g in m.GOODS];s['deck']=pool
        own=o['bonus_won'];kinds=o['opponent']['bonus_kinds'];opponent=[]
        for k,values in m.DATA['bonus_token_multisets'].items():
            pool=remaining(values,[v['value'] for v in own if v['kind']==k],rng)
            for value in take(pool,kinds.count(k)):opponent.append(dict(kind=k,value=value))
            s['bonus_tokens'][k]=pool
        # Preserve the visible chronological kind sequence, not sampled order.
        s['bonus_won'][other]=[next(opponent.pop(i) for i,v in enumerate(opponent) if v['kind']==k) for k in kinds]
    elif game=='splendor':
        slots=[]
        for e in o['history']:
            if e['player']!=other:continue
            a=e['action']
            if a['type']=='reserve':slots.append(dict(card=a.get('card'),level=a.get('level')))
            elif a['type']=='buy' and a['source']=='reserved':
                index=next((i for i,x in enumerate(slots) if x['card']==a['card']),None)
                if index is None:index=next(i for i,x in enumerate(slots) if x['card'] is None and x['level']==m.CARDS[a['card']]['level'])
                slots.pop(index)
            if 'revealed_reserved' in e:slots=[dict(card=c,level=None) for c in e['revealed_reserved']]
        if len(slots)!=o['reserved_counts'][other]:raise ValueError('Splendor reservation history')
        known=[x['card'] for x in slots if x['card'] is not None]
        used=[c for r in o['market']+o['bought'] for c in r]+o['reserved']+known
        decks=[]
        for level in (1,2,3):
            pool=remaining([c for c,d in m.CARDS.items() if d['level']==level],
                           [c for c in used if m.CARDS[c]['level']==level],rng)
            for slot in slots:
                if slot['card'] is None and slot['level']==level:slot['card']=take(pool,1)[0]
            decks.append(pool)
        s['decks']=decks;s['reserved'][other]=[x['card'] for x in slots]
        s['removed_nobles']=sorted(set(m.NOBLES)-set(o['nobles'])-set(c for a in o['acquired_nobles'] for c in a))
    elif game=='gizmos':
        used=[c for r in o['market']+o['built']+o['archives'] for c in r]
        research=o['research']
        if research:used+=research.get('cards',[])
        removed=['g0-2','g0-3'];decks=[]
        for level in (1,2,3):
            pool=remaining([c for c,d in m.CARDS.items() if d['level']==level],
                           [c for c in used if m.CARDS[c]['level']==level],rng)
            if research and research['level']==level and 'cards' not in research:
                s['research']['cards']=take(pool,research['count'])
            decks.append(take(pool,o['deck_counts'][level-1]));removed+=pool
        s['decks']=decks;s['removed']=removed
        s['internal']=[13-sum(p[i] for p in o['energy'])-o['energy_row'].count(i) for i in range(4)]
    elif game=='century_a_new_world':
        used=o['market']+[c for p in o['players'] for c in p['cards']]
        top=o['deck_top']
        if top is not None:used.append(top)
        s['deck']=([top] if top is not None else [])+remaining(m.CARDS,[c for c in used if c is not None],rng)
        used=[c for p in o['players'] for c in p['bonuses']]
        used += [b['top'] for b in o['bonus_stacks'] if b['top'] is not None]
        pool=remaining(m.BONUSES,used,rng);stacks=[]
        for b in o['bonus_stacks']:
            stacks.append(([b['top']]+take(pool,b['count']-1)) if b['count'] else [])
        s['bonus_stacks']=stacks;s['removed_bonuses']=pool
    elif game=='stone_age':
        known=[e['card'] for e in o['public_history'] if e['type']=='card_acquired' and e['player']==other]
        used=[c for c in o['market'] if c is not None]+o['players'][seat]['cards']+known
        pool=remaining(m.CARDS,used,rng)
        s['players'][other]['cards']=known+take(pool,o['players'][other]['card_count']-len(known))
        s['card_deck']=pool
        used=[c for p in o['players'] for c in p['buildings']]+[c for c in o['buildings'] if c is not None]
        pool=remaining(m.BUILDINGS,used,rng);stacks=[]
        for top,size in zip(o['buildings'],o['building_stack_sizes']):
            stacks.append(take(pool,size-1)+[top] if size else [])
        s['building_stacks']=stacks;s['removed_buildings']=pool
    elif game=='targi':
        for kind,catalog in (('tribe',m.TRIBES),('goods',m.GOODS)):
            used=list(o[kind+'_discard'])
            used += [c['card'] for c in o['center'] if c['kind']==kind and c['face_up'] and c['card'] is not None]
            if kind=='tribe':
                used += [c for p in o['players'] for r in p['rows'] for c in r]
                own=o['players'][seat]['reserve']
                if own is not None:used.append(own)
                pending=o['pending']
                if pending and pending.get('card') is not None and pending['card']!=own:used.append(pending['card'])
            pool=remaining(catalog,used,rng)
            for i,c in enumerate(o['center']):
                if c['kind']==kind and not c['face_up']:
                    s['center'][i]['card']=take(pool,1)[0] if pool else None
            if kind=='tribe' and o['players'][other]['has_reserve']:
                s['players'][other]['reserve']=take(pool,1)[0]
                s['players'][other]['reserve_round']=max(e['round'] for e in o['history'] if e['player']==other and e['action']['type']=='reserve')
            s[kind+'_deck']=pool
    elif game=='dominion':
        counts=dominion_zone_counts(o)
        for p,view in enumerate(o['players']):
            owned=Counter(copper=7,estate=3)
            for e in o['history']:
                if e['player']==p and e['event'] in ('gain','trash'):
                    owned[e['card']]+=1 if e['event']=='gain' else -1
            known=counts[p]['known']
            used=list(view['in_play'])+(list(view['hand']) if p==seat else list(known['hand'].elements()))
            top=view['discard_top']
            discard=known['discard'].copy()
            if top is not None:discard[top]=max(1,discard[top])
            used+=list(discard.elements())+list(known['deck'].elements())
            pool=remaining(owned.elements(),used,rng)
            if p!=seat:s['players'][p]['hand']=list(known['hand'].elements())+take(pool,view['hand_count']-sum(known['hand'].values()))
            s['players'][p]['deck']=list(known['deck'].elements())+take(pool,counts[p]['deck']-sum(known['deck'].values()))
            rng.shuffle(s['players'][p]['deck'])
            if top is not None:discard[top]-=1
            s['players'][p]['discard']=pool+list(discard.elements())+([top] if top is not None else [])
    elif game=='el_dorado':
        for p,view in enumerate(o['players']):
            owned=list(m.STARTER_IDS[p])
            for e in o['history']:
                if e.get('player')!=p:continue
                if e['event']=='acquire':owned.append(e['card'])
                elif e['event']=='trash' or e['event']=='play' and e.get('removed'):owned.remove(e['card'])
            used=list(view['discard'])+list(view['used'])+(view['hand'] if p==seat else [])
            pool=remaining(owned,used,rng)
            if p!=seat:s['players'][p]['hand']=take(pool,view['hand_count'])
            s['players'][p]['deck']=pool
        # Market physical IDs are deterministic by type/count, not a hidden draw.
        s['market']={name:list(m.MARKET_IDS[name][:n]) for name,n in o['market'].items()}
        ranks={b['rank'] for b in o['blockades'] if b is not None}
        ranks.update(r for p in o['players'] for r in p['blockades'])
        s['unused_blockades']=[copy_data(b) for b in m.BLOCKADES if b['rank'] not in ranks]
    elif game=='toy_battle':
        for p in (0,1):
            used=[x['uid'] for stack in o['board'].values() for x in stack if x['owner']==p]
            used += [x['uid'] for x in o['discard'][p]]
            if p==seat:used += [x['uid'] for x in o['hand']]
            pool=remaining(range(24),used,rng)
            if p!=seat:s['hands'][p]=take(pool,o['hand_sizes'][p])
            s['reserves'][p]=take(pool,o['reserve_sizes'][p]);s['removed'][p]=pool
    else:raise ValueError('No verified public sampler for '+game)


def sampled_world(env,seat,seed,*,verify=False):
    """Trusted bridge; returned world contains no actual hidden random payload."""
    if seat!=env.current_player:raise ValueError('Only the deciding player may request a world')
    game=env.game_id;obs=env.observe(seat)
    if game=='harmonies':
        from boardbench.environments.harmonies_duo import FastHarmoniesDuo
        result=game_class(game).__new__(game_class(game))
        sampled=FastHarmoniesDuo.from_observation(obs,seed)
        from .games.harmonies import _FamilyEngine
        result.engine=_FamilyEngine.__new__(_FamilyEngine)
        result.engine.load_state(sampled.save_state())
    else:
        result=game_class(game).__new__(game_class(game))
        result.state=copy_data(env.state)
        result.rng=random.Random(seed)
        _resample(result.state,obs,game,result.rng)
    if verify:
        result.validate()
        if result.observe(seat)!=obs:raise ValueError('Sampled world changes the deciding player observation')
        if result.legal_actions()!=env.legal_actions():raise ValueError('Sampled world changes legal root actions')
    return result
