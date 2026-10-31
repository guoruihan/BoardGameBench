"""Original Targi, all 45 tribe cards, two players and no expansion.

Independent rules implementation. Choices include placement, action order,
payment, table position, trades, reserved cards and immediate card abilities.
"""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product

from boardbench.families.base import DuoGame
from boardbench.families.data.targi import TRIBES, GOODS, BORDER, COORDS, OPPOSITE, RESOURCE_NAMES


def displayed(player):
    return [c for row in player['rows'] for c in row]


def effects(player):
    return {TRIBES[c]['effect'] for c in displayed(player)}


def intersections(pawns):
    rows = {COORDS[b][0] for b in pawns if COORDS[b][1] in (0, 4)}
    cols = {COORDS[b][1] for b in pawns if COORDS[b][0] in (0, 4)}
    return sorted((r-1)*3+c-1 for r in rows for c in cols)


def allocations(n, limits=None):
    for a in range(n+1):
        for b in range(n-a+1):
            v = [a, b, n-a-b]
            if limits is None or all(x <= y for x, y in zip(v, limits)):
                yield v


def payments(player, card, source):
    """Enumerate every payable alternative and every ordinary-good discount.

    The original printing/author FAQ permits ANY ordinary good for the hand
    surcharge. Gold bank is restricted to tribe costs, never raids/trading.
    """
    d = TRIBES[card]
    es = effects(player)
    discount = int('discount_'+d['kind'] in es) + int(source == 'noble' and 'noble_discount' in es)
    out = set()
    for base in d['costs']:
        costs = {tuple(base)}
        for _ in range(min(discount, sum(base[:3]))):
            costs = {tuple(x-1 if j == i else x for j, x in enumerate(c))
                     for c in costs for i in range(3) if c[i]}
        if source == 'surcharge':
            costs = {tuple(x+1 if j == i else x for j, x in enumerate(c))
                     for c in costs for i in range(3)}
        for c in costs:
            bank = min(c[3], player['bank_gold'])
            pay = c[:3]+(c[3]-bank,)
            if all(x <= y for x,y in zip(pay, player['resources'])):
                out.add((pay, bank))
    return sorted(out)


def rearrangements(rows):
    """Any final no-gap layout relocating at most two physical cards.

    This includes swaps, moving one or two row ends, and filling a temporary
    hole with the other card. Intermediate gaps are not a rules restriction.
    """
    flat = [row+[None]*(4-len(row)) for row in rows]
    old = [x for row in flat for x in row]
    seen = {tuple(old)}
    yield deepcopy(rows)
    occupied = [i for i,x in enumerate(old) if x is not None]
    for n in (1, 2):
        for src in combinations(occupied, n):
            base = list(old)
            for i in src: base[i] = None
            free = [i for i,x in enumerate(base) if x is None]
            for dst in product(free, repeat=n):
                if len(set(dst)) != n: continue
                candidate = list(base)
                for i,j in zip(src,dst): candidate[j] = old[i]
                key = tuple(candidate)
                if key in seen: continue
                valid = True
                for r in range(3):
                    part = candidate[4*r:4*r+4]
                    if None in part and any(x is not None for x in part[part.index(None):]):
                        valid = False; break
                if valid:
                    seen.add(key)
                    yield [[c for c in candidate[4*r:4*r+4] if c is not None] for r in range(3)]


def breakdown(player, opponent):
    cards = displayed(player)
    es = effects(player)
    counts = Counter(TRIBES[c]['kind'] for c in cards)
    same = diverse = 0
    for row in player['rows']:
        if len(row) == 4:
            kinds = {TRIBES[c]['kind'] for c in row}
            same += len(kinds) == 1
            diverse += len(kinds) == 4
    bonus = sum(counts[e[6:]]//2 for e in es if e.startswith('pairs_'))
    bonus += int('majority' in es and len(cards) > len(displayed(opponent)))
    bonus += int('ten_cards' in es and len(cards) >= 10)
    bonus += 2*min(2, diverse) if 'diverse_rows' in es else 0
    bonus += sum(2 for row in player['rows'] if len(row) == 4 and TRIBES[row[-1]]['effect'] == 'rightmost')
    result = dict(tokens=player['vp'], cards=sum(TRIBES[c]['vp'] for c in cards),
                  rows=4*same+2*diverse, abilities=bonus)
    result['total'] = sum(result.values())
    return result


class Targi(DuoGame):
    game_id = 'targi'
    config_id = 'original_2012_duo'

    def _reset(self, first_player):
        players = [dict(resources=[2,2,2,1], vp=4, rows=[[],[],[]], reserve=None,
                        reserve_round=None, bank_gold=0, pawns=[], markers=[],
                        received=[False]*3, bonus_used=[False]*3, hand_used=False,
                        penalty_round=None, quota=3) for _ in (0,1)]
        td, gd = list(TRIBES), list(GOODS)
        self.rng.shuffle(td); self.rng.shuffle(gd)
        self.state.update(players=players, tribe_deck=td, goods_deck=gd,
                          tribe_discard=[], goods_discard=[], center=[],
                          phase='place', round=1, robber=1, pending=None,
                          bonus_count=0, bonus_return='actions', raid_done=[], history=[])
        for i in range(9):
            kind = 'goods' if i % 2 == 0 else 'tribe'
            self.state['center'].append(dict(kind=kind, card=self._draw(kind), face_up=True))

    def _draw(self, kind):
        s = self.state
        deck, discard = s[kind+'_deck'], s[kind+'_discard']
        # Goods reshuffle is immediate, including after a last-card draw.
        # Tribe exhaustion uses the same discard reshuffle; see coverage note.
        if not deck and discard:
            deck.extend(discard); discard.clear(); self.rng.shuffle(deck)
        if not deck: return None
        card = deck.pop()
        if not deck and discard:
            deck.extend(discard); discard.clear(); self.rng.shuffle(deck)
        return card

    def _build_actions(self, card, source):
        p = self.state['players'][self.current_player]
        return [dict(type='build', card=card, source=source, row=r, payment=list(pay), bank=bank)
                for pay,bank in payments(p,card,source)
                for r,row in enumerate(p['rows']) if len(row)<4]

    def _legal_actions(self):
        s = self.state; p = s['players'][self.current_player]; q = s['players'][1-self.current_player]
        phase = s['phase']; es = effects(p)
        if phase == 'place':
            forbidden = set(p['pawns']+q['pawns']) | {OPPOSITE[x] for x in q['pawns']}
            if 'robber_placement' not in es: forbidden.add(s['robber'])
            return [dict(type='place', border=b) for b in OPPOSITE if b not in forbidden]
        if phase == 'actions':
            out = [dict(type=t, zone=z, position=i) for z,ids in [('border',p['pawns']),('center',p['markers'])]
                   for i in ids for t in ('use','skip')]
            if 'hand_surcharge' in es and not p['hand_used'] and p['reserve'] is not None:
                out += self._build_actions(p['reserve'],'surcharge')
            if not p['pawns'] and not p['markers']: out.append(dict(type='finish_turn'))
            return out
        if phase == 'tribe':
            card = s['pending']['card']; source = s['pending']['source']
            out = [dict(type='decline')]
            if card is None: return out
            if source != 'noble' or p['reserve_round'] < s['round']:
                out += self._build_actions(card, source)
            if source == 'noble': out.append(dict(type='discard_reserve'))
            elif p['reserve'] is None: out.append(dict(type='reserve'))
            return out
        if phase == 'choose_goods':
            return [dict(type='gain_goods', goods=x) for x in allocations(s['bonus_count'])]
        if phase == 'choice_bonus':
            return [dict(type='bonus', choice=x) for x in ('goods','gold','vp')]
        if phase == 'rearrange':
            return [dict(type='rearrange',rows=rows) for rows in rearrangements(p['rows'])]
        if phase == 'trader':
            out = [dict(type='done')]
            rate = 1 if 'trader_discount' in es else 2
            for i in range(3):
                if p['resources'][i] >= 3: out.append(dict(type='trade',give=i,take=3,rate=3))
                if p['resources'][i] >= rate:
                    out += [dict(type='trade',give=i,take=j,rate=rate) for j in range(3) if j!=i]
            return out
        if phase == 'silversmith':
            out = [dict(type='done')]
            for i in range(4):
                for n,v in ((1,2),(2,4)) if i==3 else ((2,1),(4,3)):
                    if p['resources'][i]>=n: out.append(dict(type='sell',resource=i,count=n,vp=v))
            return out
        if phase == 'fata_morgana':
            out = [dict(type='done')]; occupied = set(p['markers']+q['markers'])
            for i in p['markers']:
                out += [dict(type='move_marker',source=i,target=j) for j,c in enumerate(s['center'])
                        if j not in occupied and c['face_up'] and c['card'] is not None]
            return out
        if phase == 'discard_excess':
            n = max(0,sum(p['resources'][:3])-10)
            return [dict(type='discard_goods', goods=x) for x in allocations(n,p['resources'][:3])]
        if phase == 'raid':
            out = []
            if 'raid_immunity' in es: out.append(dict(type='raid_immune'))
            n,v = {4:(1,1),8:(2,1),12:(3,2),16:(0,3)}[s['robber']]
            if p['vp']>=v: out.append(dict(type='raid_vp',vp=v))
            if n:
                out += [dict(type='raid_goods',goods=x) for x in allocations(n,p['resources'][:3])]
            elif p['resources'][3]: out.append(dict(type='raid_gold'))
            if not out: out.append(dict(type='raid_insolvent',opponent_vp=v))
            return out
        raise AssertionError(phase)

    def _gain(self, p, vector, goods_card=False):
        player = self.state['players'][p]
        for i,n in enumerate(vector[:4]): player['resources'][i] += n
        if len(vector)>4: player['vp'] += vector[4]
        if goods_card:
            for i in range(3): player['received'][i] |= vector[i]>0
            self._goods_bonuses(p)

    def _goods_bonuses(self, p):
        player = self.state['players'][p]; es = effects(player)
        for i,key in enumerate(('dates','salt','pepper')):
            if player['received'][i] and not player['bonus_used'][i] and 'bonus_'+key in es:
                player['resources'][i] += 1; player['bonus_used'][i] = True

    def _choice(self,n,return_to='actions',goods_card=False):
        if n:
            self.state.update(phase='choose_goods',bonus_count=n,bonus_return=return_to,
                              choice_is_goods_card=goods_card)
        else: self.state['phase'] = return_to

    def _consume(self,position):
        s = self.state; old = s['center'][position]
        kind = 'tribe' if old['kind']=='goods' else 'goods'
        s['center'][position] = dict(kind=kind,card=self._draw(kind),face_up=False)
        return old

    def _acquire_goods(self,card):
        s = self.state
        if card is None: return
        s['goods_discard'].append(card)
        if GOODS[card] is None: self._choice(1,goods_card=True)
        else: self._gain(self.current_player,GOODS[card],goods_card=True)

    def _build(self,a):
        s = self.state; p = s['players'][self.current_player]; card=a['card']
        for i,n in enumerate(a['payment']): p['resources'][i] -= n
        p['bank_gold'] -= a['bank']; p['rows'][a['row']].append(card)
        if a['source'] in ('noble','surcharge'): p['reserve']=None;p['reserve_round']=None
        if a['source']=='surcharge': p['hand_used']=True
        s.update(pending=None,phase='actions')
        effect = TRIBES[card]['effect']; count=len(displayed(p))
        if effect=='bank_gold': p['bank_gold'] += 1
        elif effect=='two_pawns': s['players'][1-self.current_player]['penalty_round']=s['round']+1
        elif effect=='display_goods': self._choice(count//2)
        elif effect=='empty_goods': self._choice((12-count)//2)
        elif effect=='choice_bonus': s['phase']='choice_bonus'
        elif effect=='rearrange': s['phase']='rearrange'
        self._goods_bonuses(self.current_player)

    def _step(self,a):
        s=self.state; actor=self.current_player; p=s['players'][actor]; t=a['type']
        # Only actual public actions enter history; hidden draws/hand identities
        # are revealed only if later built or publicly discarded.
        event=deepcopy(a)
        if t=='build': event['card']=a['card']
        s['history'].append(dict(player=actor,round=s['round'],action=event))
        if t=='place':
            p['pawns'].append(a['border'])
            other=s['players'][1-actor]
            if len(other['pawns'])<other['quota']: s['current_player']=1-actor
            elif len(p['pawns'])==p['quota']:
                for pl in s['players']: pl['markers']=intersections(pl['pawns'])
                s.update(phase='actions',current_player=s['first_player'])
        elif t in ('use','skip'):
            ids=p['pawns'] if a['zone']=='border' else p['markers'];ids.remove(a['position'])
            if t=='skip': return  # Unused central card remains, author FAQ #5.
            if a['zone']=='center':
                old=self._consume(a['position'])
                if old['kind']=='goods': self._acquire_goods(old['card'])
                else: s.update(phase='tribe',pending=dict(card=old['card'],source='center'))
            else:
                effect=BORDER[a['position']]
                if effect in RESOURCE_NAMES:
                    p['resources'][RESOURCE_NAMES.index(effect)]+=1
                elif effect=='caravan': self._acquire_goods(self._draw('goods'))
                elif effect=='tribal_expansion': s.update(phase='tribe',pending=dict(card=self._draw('tribe'),source='expansion'))
                elif effect=='noble':
                    if p['reserve'] is not None:s.update(phase='tribe',pending=dict(card=p['reserve'],source='noble'))
                else: s['phase']=effect
        elif t=='build': self._build(a)
        elif t in ('decline','reserve','discard_reserve'):
            pending=s['pending'];card=pending['card']
            if t=='reserve':p['reserve']=card;p['reserve_round']=s['round']
            elif t=='discard_reserve':s['tribe_discard'].append(card);p['reserve']=None;p['reserve_round']=None
            elif pending['source']!='noble' and card is not None:s['tribe_discard'].append(card)
            s.update(phase='actions',pending=None)
        elif t=='gain_goods':
            self._gain(actor,a['goods']+[0],goods_card=s.get('choice_is_goods_card',False))
            s['phase']=s['bonus_return'];s['bonus_count']=0
            if s['phase']=='end_bonus_done': self._end_limit()
        elif t=='bonus':
            if a['choice']=='goods': self._choice(3)
            else:
                if a['choice']=='gold':p['resources'][3]+=1
                else:p['vp']+=1
                s['phase']='actions'
        elif t=='rearrange':p['rows']=deepcopy(a['rows']);s['phase']='actions'
        elif t=='trade':p['resources'][a['give']]-=a['rate'];p['resources'][a['take']]+=1
        elif t=='sell':p['resources'][a['resource']]-=a['count'];p['vp']+=a['vp'];s['phase']='actions'
        elif t=='done':s['phase']='actions'
        elif t=='move_marker':p['markers'].remove(a['source']);p['markers'].append(a['target']);p['markers'].sort();s['phase']='actions'
        elif t=='finish_turn':
            if 'no_goods' in effects(p) and not any(p['received']):self._choice(1,'end_bonus_done')
            else:self._end_limit()
        elif t=='discard_goods':
            for i,n in enumerate(a['goods']):p['resources'][i]-=n
            self._end_player()
        elif t.startswith('raid_'):
            if t=='raid_vp':p['vp']-=a['vp']
            elif t=='raid_goods':
                for i,n in enumerate(a['goods']):p['resources'][i]-=n
            elif t=='raid_gold':p['resources'][3]-=1
            elif t=='raid_insolvent':s['players'][1-actor]['vp']+=a['opponent_vp']
            if t in ('raid_vp','raid_goods','raid_gold') and 'raid_point' in effects(p):p['vp']+=1
            s['raid_done'].append(actor)
            if len(s['raid_done'])<2:s['current_player']=1-actor
            elif s['robber']==16:self._finish_game('fourth_raid')
            else:s['robber']+=1;s.update(phase='place',current_player=s['first_player'])
        else:raise AssertionError(t)

    def _end_limit(self):
        p=self.state['players'][self.current_player]
        p['resources'][3]=min(3,p['resources'][3])
        if sum(p['resources'][:3])>10:self.state['phase']='discard_excess'
        else:self._end_player()

    def _end_player(self):
        s=self.state
        if self.current_player==s['first_player']:
            s.update(current_player=1-self.current_player,phase='actions');return
        if any(len(displayed(p))==12 for p in s['players']):
            self._finish_game('twelve_tribes');return
        for c in s['center']:c['face_up']=True
        s['first_player']=1-s['first_player'];s['round']+=1;s['robber']+=1
        s.update(current_player=s['first_player'],phase='place',raid_done=[])
        for p in s['players']:
            p.update(received=[False]*3,bonus_used=[False]*3,hand_used=False,
                     quota=2 if p['penalty_round']==s['round'] else 3)
        if s['robber']%4==0:s['phase']='raid'

    def _finish_game(self,reason):
        ps=self.state['players'];sc=self.scores()
        keys=[(sc[i],p['resources'][3],sum(p['resources'][:3])) for i,p in enumerate(ps)]
        self.finish([i for i,k in enumerate(keys) if k==max(keys)],reason)

    def scores(self):
        return [x['total'] for x in self.score_breakdown()]

    def score_breakdown(self):
        ps=self.state['players']
        return [breakdown(ps[i],ps[1-i]) for i in (0,1)]

    def _observe(self,player):
        s=self.state; ps=[]
        for i,p in enumerate(s['players']):
            public=deepcopy(p)
            if i!=player:public['reserve']=None;public['reserve_round']=None
            public['has_reserve']=p['reserve'] is not None
            ps.append(public)
        center=[dict(kind=c['kind'],card=c['card'] if c['face_up'] else None,face_up=c['face_up']) for c in s['center']]
        pending=deepcopy(s['pending'])
        if pending and player!=s['current_player'] and pending['source']=='noble':pending['card']=None
        return dict(players=ps,center=center,phase=s['phase'],round=s['round'],robber=s['robber'],
                    first_player=s['first_player'],pending=pending,bonus_count=s['bonus_count'],
                    tribe_deck_count=len(s['tribe_deck']),goods_deck_count=len(s['goods_deck']),
                    tribe_discard=list(s['tribe_discard']),goods_discard=list(s['goods_discard']),
                    history=deepcopy(s['history']),score_breakdown=self.score_breakdown())

    def validate(self):
        super().validate();s=self.state
        tribes=list(s['tribe_deck'])+list(s['tribe_discard']); goods=list(s['goods_deck'])+list(s['goods_discard'])
        for c in s['center']:
            if c['card'] is not None:(tribes if c['kind']=='tribe' else goods).append(c['card'])
        for p in s['players']:
            tribes.extend(displayed(p))
            if p['reserve'] is not None:tribes.append(p['reserve'])
            if len(p['rows'])!=3 or any(len(r)>4 for r in p['rows']):raise ValueError('invalid tribe layout')
            if len(p['pawns'])!=len(set(p['pawns'])) or len(p['markers'])!=len(set(p['markers'])):raise ValueError('duplicate figures')
            if any(type(v) is not int or v<0 for v in p['resources']+[p['vp'],p['bank_gold']]):raise ValueError('invalid resources')
        pending=s['pending']
        if pending and pending['card'] is not None and pending['source']!='noble':tribes.append(pending['card'])
        if sorted(tribes)!=list(TRIBES) or sorted(goods)!=list(GOODS):raise ValueError('card conservation failure')
        if not 1<=s['robber']<=16:raise ValueError('invalid robber')

    @staticmethod
    def features(observation,action):
        o=observation;p=o['players'][o['player']];t=action['type'];r=p['resources'];es=effects(p)
        f=[0.]*24;f[1]=o['round']/12;f[2]=sum(r[:3])/10;f[3]=r[3]/3;f[4]=len(displayed(p))/12
        def goods_value(v):
            return sum(n*(.75 if i==3 else .25+.45/(1+r[i])) for i,n in enumerate(v))
        def card_value(card):
            d=TRIBES[card];cost=min(sum(c[:3])*.45+c[3]*1.1 for c in d['costs'])
            engine=1.2*max(0,12-o['round'])/12 if d['effect'] else 0
            return d['vp']+.8+engine-.35*cost
        def center_value(i):
            c=o['center'][i]
            if c['card'] is None:return 0.
            if c['kind']=='goods':return .55 if GOODS[c['card']] is None else goods_value(GOODS[c['card']][:4])+GOODS[c['card']][4]
            return card_value(c['card']) if payments(p,c['card'],'center') else .35
        def border_value(b):
            name=BORDER[b]
            if name in RESOURCE_NAMES:return goods_value([int(i==RESOURCE_NAMES.index(name)) for i in range(4)])
            if name=='noble':return card_value(p['reserve']) if p['reserve'] is not None and payments(p,p['reserve'],'noble') else 0
            if name=='trader':return .6 if max(r[:3])>=3 and r[3]<2 else .1
            if name=='silversmith':return max([v-.45*n for i in range(3) for n,v in [(2,1),(4,3)] if r[i]>=n]+[.5 if r[3] else 0])
            return {'caravan':.7,'tribal_expansion':.7,'fata_morgana':.4}.get(name,0.)
        if t=='place':
            before=set(intersections(p['pawns']));after=set(intersections(p['pawns']+[action['border']]))
            f[5]=border_value(action['border']);f[6]=sum(center_value(i) for i in after-before)
            # A first pawn needs potential over future perpendicular lines.
            if not after:
                row,col=COORDS[action['border']]
                potentials=[center_value((i-1)*3+col-1) for i in range(1,4)] if row in (0,4) else [center_value((row-1)*3+i-1) for i in range(1,4)]
                f[7]=max(potentials)*.45
            f[0]=f[5]+f[6]+f[7]
        elif t=='use':
            f[8]=center_value(action['position']) if action['zone']=='center' else border_value(action['position'])
            if action['zone']=='border' and BORDER[action['position']]=='trader':f[8]-=.3
            f[0]=1+f[8]
        elif t=='build':
            d=TRIBES[action['card']];row=p['rows'][action['row']];kinds=[TRIBES[c]['kind'] for c in row]+[d['kind']]
            f[9]=d['vp'];f[10]=goods_value(action['payment']);f[11]=card_value(action['card'])
            f[12]=1 if len(set(kinds))==1 else .5 if len(set(kinds))==len(kinds) else -.3
            f[13]=len(row)/4;f[14]=1 if d['effect']=='rightmost' and len(row)==3 else 0
            f[0]=3+f[11]-.4*f[10]+f[12]+f[13]+2*f[14]
        elif t=='reserve':f[0]=.25
        elif t=='gain_goods':f[0]=goods_value(action['goods'])
        elif t=='bonus':f[0]={'goods':.9,'gold':1.0 if r[3]<2 else .5,'vp':.7}[action['choice']]
        elif t=='trade':
            f[15]=(1 if action['take']==3 else 0);f[16]=max(0,r[action['give']]-2)
            f[17]=max(0,2-r[action['take']]);f[0]=-.2+f[15]*.4+f[16]*.12+f[17]*.1
            if action['take']==3 and r[3]>=2:f[0]-=1
            if action['take']<3 and r[action['give']]<=r[action['take']]+1:f[0]-=1
        elif t=='sell':f[18]=action['vp'];f[0]=action['vp']-.35*action['count']-(.5 if action['resource']==3 else 0)
        elif t=='move_marker':f[0]=center_value(action['target'])-center_value(action['source'])
        elif t=='rearrange':
            candidate=deepcopy(p);candidate['rows']=action['rows']
            q=o['players'][1-o['player']];f[0]=breakdown(candidate,q)['total']-breakdown(p,q)['total']
            f[0]+=sum(.2*len(row) for row in candidate['rows'] if row and len({TRIBES[c]['kind'] for c in row})==1)
        elif t=='discard_goods':f[0]=-goods_value(action['goods'])
        elif t=='raid_immune':f[0]=2
        elif t=='raid_goods':f[0]=-goods_value(action['goods'])
        elif t=='raid_gold':f[0]=-.9
        elif t=='raid_vp':f[0]=-action['vp']
        elif t=='raid_insolvent':f[0]=-action['opponent_vp']
        elif t in ('skip','decline','discard_reserve'):f[0]=-1
        f[19]=float('no_goods' in es);f[20]=float('trader_discount' in es);f[21]=float('hand_surcharge' in es)
        f[22]=float(p['reserve'] is not None);f[23]=float(o['player']==o['first_player'])
        return f


GAME_CLASS = Targi
