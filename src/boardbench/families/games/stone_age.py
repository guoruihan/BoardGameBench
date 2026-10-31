"""Classic Stone Age, genuine 36-card/28-building base set, two players."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/stone_age.json').read_text())
CARDS = {c['id']: c for c in DATA['cards']}
BUILDINGS = {b['id']: b for b in DATA['buildings']}
RESOURCES = tuple(DATA['resources'])
VALUES = tuple(DATA['resource_values'])
VILLAGE = ('tool', 'hut', 'field')
GATHER = ('hunt', 'wood', 'clay', 'stone', 'gold')
LOCATIONS = VILLAGE + GATHER + tuple('card'+str(i) for i in range(4)) + ('building0','building1')


@lru_cache(maxsize=16384)
def payments(resources, minimum, maximum, kinds=None):
    """All affordable resource multisets, never heuristic-pruned."""
    result = []
    for w in range(min(resources[0], maximum)+1):
        for c in range(min(resources[1], maximum-w)+1):
            for s in range(min(resources[2], maximum-w-c)+1):
                for g in range(max(0,minimum-w-c-s), min(resources[3],maximum-w-c-s)+1):
                    v = (w,c,s,g)
                    if kinds is None or sum(x>0 for x in v) == kinds:
                        result.append(v)
    return tuple(result)


def building_payments(resources, building):
    if building['kind'] == 'fixed':
        cost = tuple(building['cost'])
        return (cost,) if all(a>=b for a,b in zip(resources,cost)) else ()
    return payments(tuple(resources), building['minimum'], building['maximum'], building['kinds'])


def card_totals(cards):
    culture = Counter(); mult = Counter()
    for cid in cards:
        b = CARDS[cid]['bottom']
        if b['kind'] == 'culture': culture[b['symbol']] += 1
        else: mult[b['stat']] += b['count']
    return culture, mult


def final_breakdown(player):
    culture, mult = card_totals(player['cards'])
    sets = [sum(n>=i for n in culture.values()) for i in range(1,max(culture.values(),default=0)+1)]
    multipliers = dict(farmers=mult['farmers']*player['agriculture'],
                       toolmakers=mult['toolmakers']*sum(t['value'] for t in player['tools']),
                       builders=mult['builders']*len(player['buildings']),
                       shamans=mult['shamans']*player['population'])
    total = player['points'] + sum(v*v for v in sets) + sum(multipliers.values()) + sum(player['resources'])
    return dict(points=player['points'], culture_sets=sets, culture=sum(v*v for v in sets),
                multipliers=multipliers, resources=sum(player['resources']), total=total)


def public_card_value(player, cid):
    """Action heuristic based exclusively on the actor's own known cards."""
    culture, mult = card_totals(player['cards']); card = CARDS[cid]; b = card['bottom']
    if b['kind'] == 'culture':
        layer = culture[b['symbol']]+1
        count = sum(n>=layer for n in culture.values())
        value = 2*count+1 + (3 if layer==1 else .5)
    else:
        stat = dict(farmers=max(player['agriculture'],3),
                    toolmakers=max(sum(t['value'] for t in player['tools']),3),
                    builders=max(len(player['buildings']),2),shamans=player['population'])[b['stat']]
        value = b['count']*stat
    e = card['effect']; typ = e['type']
    value += {'dice_draft':3.5,'food':min(e.get('amount',0),max(0,player['population']-player['agriculture'])),
              'resource':2*e.get('amount',0),'roll':3,'points':3,'agriculture':5,
              'tool':5,'one_use_tool':e.get('amount',0)*.7,'deferred_resources':5,'extra_card':7}.get(typ,0)
    return value


class StoneAge(DuoGame):
    game_id = 'stone_age'
    rules_version = 'classic_2008_2014'
    config_id = 'classic_base_duo'

    def _reset(self, first_player):
        cards = list(CARDS); buildings = list(BUILDINGS)
        self.rng.shuffle(cards); self.rng.shuffle(buildings)
        players = [dict(population=5, agriculture=0, tools=[], food=12, resources=[0]*4,
                        cards=[], buildings=[], points=0, one_use=[], deferred=[], unplaced=5)
                   for _ in (0,1)]
        self.state.update(players=players, card_deck=cards,
                          market=[cards.pop() for _ in range(4)],
                          building_stacks=[buildings[:7],buildings[7:14]], removed_buildings=buildings[14:],
                          placements={loc:[0,0] for loc in LOCATIONS}, phase='place', round=1,
                          resolution_order=[first_player,1-first_player], resolver=0,
                          effects=[], pending=None, interrupt=None, feeding_index=0,
                          public_history=[], produced=[24,0,0,0,0], spent=[0]*5, final_scores=None)

    def _gain(self, p, resource, amount):
        s=self.state; i=0 if resource=='food' else RESOURCES.index(resource)+1
        if i==0: s['players'][p]['food'] += amount
        else: s['players'][p]['resources'][i-1] += amount
        s['produced'][i] += amount

    def _pay(self, p, cost):
        for i,n in enumerate(cost):
            self.state['players'][p]['resources'][i] -= n
            self.state['spent'][i+1] += n

    def _placement_actions(self, p):
        s=self.state; player=s['players'][p]; left=player['unplaced']; result=[]
        village_used=sum(any(s['placements'][loc]) for loc in VILLAGE)
        for loc in LOCATIONS:
            occ=s['placements'][loc]
            if occ[p] or (loc!='hunt' and any(occ)): continue
            if loc in VILLAGE and village_used>=2: continue
            if loc.startswith('card') and s['market'][int(loc[4:])] is None: continue
            if loc.startswith('building') and not s['building_stacks'][int(loc[8:])]: continue
            if loc=='hut': amounts=[2] if left>=2 else []
            elif loc in GATHER: amounts=range(1,min(left,7 if loc!='hunt' else left)+1)
            else: amounts=[1] if left else []
            result += [dict(type='place',location=loc,count=n) for n in amounts]
        return result

    def _deferred_actions(self, p):
        return [dict(type='take_two',card=cid,resources=list(v))
                for cid in self.state['players'][p]['deferred'] for v in payments((2,2,2,2),2,2)]

    def _legal_actions(self):
        s=self.state; p=s['current_player']; player=s['players'][p]; phase=s['phase']
        if phase=='resource_window':
            return [dict(type='wait')]+self._deferred_actions(p)
        result=self._deferred_actions(p)
        if phase=='place': result += self._placement_actions(p)
        elif phase=='resolve':
            result += [dict(type='resolve',location=loc) for loc in LOCATIONS if s['placements'][loc][p]]
        elif phase=='pay_card':
            cost=s['pending']['slot']+1
            result += [dict(type='decline')]+[dict(type='buy_card',resources=list(v)) for v in payments(tuple(player['resources']),cost,cost)]
        elif phase=='pay_building':
            b=BUILDINGS[s['pending']['building']]
            result += [dict(type='decline')]+[dict(type='build',resources=list(v)) for v in building_payments(player['resources'],b)]
        elif phase=='roll':
            permanent=[i for i,t in enumerate(player['tools']) if not t['used']]
            one_use=player['one_use']
            for n in range(len(permanent)+1):
                for subset in combinations(permanent,n):
                    for k in range(len(one_use)+1):
                        result += [dict(type='use_tools',permanent=list(subset),one_use=list(c)) for c in combinations(one_use,k)]
        elif phase=='upgrade':
            low=min(t['value'] for t in player['tools'])
            result += [dict(type='upgrade_tool',index=i) for i,t in enumerate(player['tools']) if t['value']==low]
        elif phase=='draft_die':
            result += [dict(type='choose_die',value=d) for d in sorted(set(s['pending']['dice']))]
        elif phase=='feed':
            deficit=max(0,player['population']-player['food']-player['agriculture'])
            if deficit:
                result += [dict(type='starve')]+[dict(type='feed',resources=list(v)) for v in payments(tuple(player['resources']),deficit,deficit)]
            else: result.append(dict(type='feed',resources=[0]*4))
        return result

    def _step(self, action):
        s=self.state; p=s['current_player']; player=s['players'][p]; typ=action['type']
        s['public_history'].append(dict(action,player=p,round=s['round']))
        if typ in ('wait','take_two'):
            if typ=='take_two':
                player['deferred'].remove(action['card'])
                for res,n in zip(RESOURCES,action['resources']): self._gain(p,res,n)
            if s['phase']=='resource_window':
                resume=s['interrupt'];s['phase']=resume['phase'];s['current_player']=resume['player'];s['interrupt']=None
            return
        if typ=='place':
            loc=action['location']; n=action['count'];player['unplaced']-=n;s['placements'][loc][p]=n
            if self._placement_actions(1-p): s['current_player']=1-p
            elif self._placement_actions(p): s['current_player']=p
            else:
                s['phase']='resolve';s['resolver']=0;s['current_player']=s['resolution_order'][0]
                self._advance_resolver()
        elif typ=='resolve':
            loc=action['location']; n=s['placements'][loc][p];s['placements'][loc][p]=0
            if loc in GATHER:
                self._queue(dict(type='roll',player=p,resource='food' if loc=='hunt' else loc,dice=n))
            elif loc=='tool': self._queue(dict(type='tool',player=p))
            elif loc=='field': self._queue(dict(type='agriculture',player=p,amount=1))
            elif loc=='hut':
                player['population']=min(10,player['population']+1);self._advance_resolver()
            elif loc.startswith('card'):
                i=int(loc[4:]);s['pending']=dict(slot=i,card=s['market'][i]);s['phase']='pay_card'
            else:
                i=int(loc[8:]);s['pending']=dict(slot=i,building=s['building_stacks'][i][-1]);s['phase']='pay_building'
        elif typ=='decline':
            s['pending']=None;s['phase']='resolve';self._advance_resolver()
        elif typ=='buy_card':
            pending=s['pending'];cid=pending['card'];self._pay(p,action['resources']);s['market'][pending['slot']]=None
            player['cards'].append(cid);s['pending']=None
            s['public_history'].append(dict(type='card_acquired',player=p,card=cid))
            self._queue(dict(CARDS[cid]['effect'],player=p,card=cid))
        elif typ=='build':
            pending=s['pending'];bid=pending['building'];b=BUILDINGS[bid];self._pay(p,action['resources'])
            s['building_stacks'][pending['slot']].pop();player['buildings'].append(bid)
            points=b['points'] if b['kind']=='fixed' else sum(n*v for n,v in zip(action['resources'],VALUES))
            player['points']+=points
            nxt=s['building_stacks'][pending['slot']]
            s['public_history'].append(dict(type='building_acquired',player=p,building=bid,points=points,
                                            slot=pending['slot'],revealed=nxt[-1] if nxt else None))
            s['pending']=None;s['phase']='resolve';self._advance_resolver()
        elif typ=='use_tools':
            bonus=0
            for i in action['permanent']:
                t=player['tools'][i];bonus+=t['value'];t['used']=True
            for cid in action['one_use']:
                bonus+=CARDS[cid]['effect']['amount'];player['one_use'].remove(cid)
            effect=s['effects'].pop(0);roll=s['pending']['dice'];divisor=2 if effect['resource']=='food' else VALUES[RESOURCES.index(effect['resource'])]
            gained=(sum(roll)+bonus)//divisor;self._gain(p,effect['resource'],gained)
            s['public_history'].append(dict(type='gathered',player=p,resource=effect['resource'],amount=gained,bonus=bonus))
            s['pending']=None;self._process_effects()
        elif typ=='upgrade_tool':
            player['tools'][action['index']]['value']+=1
            s['effects'].pop(0);s['pending']=None;self._process_effects()
        elif typ=='choose_die':
            draft=s['pending'];d=action['value'];draft['dice'].remove(d);draft['next']+=1
            # Store the remaining draft as its own effect, so an upgrade choice can suspend it.
            s['effects'].pop(0)
            if draft['dice']:
                s['effects'].insert(0,dict(type='continue_draft',player=1-p,dice=draft['dice'],next=draft['next']))
            if d<=4: self._gain(p,RESOURCES[d-1],1)
            elif d==5: s['effects'].insert(0,dict(type='tool',player=p))
            else: s['effects'].insert(0,dict(type='agriculture',player=p,amount=1))
            s['pending']=None;self._process_effects()
        elif typ in ('feed','starve'):
            self._gain(p,'food',player['agriculture'])
            eaten=min(player['food'],player['population']);player['food']-=eaten;s['spent'][0]+=eaten
            if typ=='starve': player['points']-=10
            else: self._pay(p,action['resources'])
            s['feeding_index']+=1
            if s['feeding_index']==2: self._end_round()
            else: s['current_player']=s['resolution_order'][s['feeding_index']]
        self._offer_deferred_interrupt()

    def _queue(self,effect):
        self.state['effects'].append(effect);self._process_effects()

    def _process_effects(self):
        s=self.state
        while s['effects']:
            e=s['effects'][0];p=e['player'];player=s['players'][p];s['current_player']=p;typ=e['type']
            if typ=='tool':
                if len(player['tools'])<3: player['tools'].append(dict(value=1,used=False))
                elif min(t['value'] for t in player['tools'])<4:
                    s['phase']='upgrade';s['pending']=None;return
            elif typ=='agriculture': player['agriculture']=min(10,player['agriculture']+e['amount'])
            elif typ=='food': self._gain(p,'food',e['amount'])
            elif typ=='resource': self._gain(p,e['resource'],e['amount'])
            elif typ=='points': player['points']+=e['amount']
            elif typ=='one_use_tool': player['one_use'].append(e['card'])
            elif typ=='deferred_resources': player['deferred'].append(e['card'])
            elif typ=='extra_card':
                if s['card_deck']:
                    player['cards'].append(s['card_deck'].pop())
                    s['public_history'].append(dict(type='private_bonus_card',player=p))
            elif typ=='roll':
                dice=[self.rng.randint(1,6) for _ in range(e['dice'])]
                s['pending']=dict(dice=dice,resource=e['resource']);s['phase']='roll'
                s['public_history'].append(dict(type='roll',player=p,resource=e['resource'],dice=dice))
                return
            elif typ in ('dice_draft','continue_draft'):
                dice=[self.rng.randint(1,6) for _ in range(2)] if typ=='dice_draft' else e['dice']
                s['pending']=dict(dice=dice,next=e.get('next',0));s['phase']='draft_die'
                if typ=='dice_draft': s['public_history'].append(dict(type='dice_draft_rolled',player=p,dice=dice[:]))
                return
            s['effects'].pop(0)
        s['phase']='resolve';s['pending']=None;s['current_player']=s['resolution_order'][s['resolver']]
        self._advance_resolver()

    def _advance_resolver(self):
        s=self.state
        while s['resolver']<2:
            p=s['resolution_order'][s['resolver']];s['current_player']=p
            if any(occ[p] for occ in s['placements'].values()): return
            s['resolver']+=1
        s['phase']='feed';s['feeding_index']=0;s['current_player']=s['resolution_order'][0]

    def _offer_deferred_interrupt(self):
        s=self.state
        if s['terminated'] or s['phase']=='resource_window': return
        other=1-s['current_player']
        if s['players'][other]['deferred']:
            s['interrupt']=dict(phase=s['phase'],player=s['current_player'])
            s['phase']='resource_window';s['current_player']=other

    def _end_round(self):
        s=self.state
        if any(not stack for stack in s['building_stacks']): self._finish_game('building_stack_exhausted');return
        remaining=[c for c in s['market'] if c is not None]
        if len(s['card_deck'])<4-len(remaining): self._finish_game('civilization_refill_exhausted');return
        while len(remaining)<4: remaining.append(s['card_deck'].pop())
        s['market']=remaining;s['round']+=1;s['resolution_order'].reverse();s['first_player']=s['resolution_order'][0]
        s['current_player']=s['first_player'];s['phase']='place';s['resolver']=0;s['feeding_index']=0
        s['placements']={loc:[0,0] for loc in LOCATIONS}
        for player in s['players']:
            player['unplaced']=player['population']
            for t in player['tools']:t['used']=False
        s['public_history'].append(dict(type='new_round',round=s['round'],first_player=s['first_player'],market=remaining[:]))

    def _finish_game(self,reason):
        s=self.state;s['final_scores']=[final_breakdown(p)['total'] for p in s['players']]
        winners=[i for i in (0,1) if s['final_scores'][i]==max(s['final_scores'])]
        if len(winners)==2:
            tie=[p['agriculture']+sum(t['value'] for t in p['tools'])+p['population'] for p in s['players']]
            winners=[i for i in (0,1) if tie[i]==max(tie)]
        s['phase']='finished';self.finish(winners,reason)

    def scores(self):
        return list(self.state['final_scores']) if self.state['final_scores'] is not None else [p['points'] for p in self.state['players']]

    def score_breakdown(self):
        if self.state['terminated']:return [final_breakdown(p) for p in self.state['players']]
        return [dict(points=p['points'],total=p['points']) for p in self.state['players']]

    def _observe(self,player):
        s=self.state;players=[]
        for i,p in enumerate(s['players']):
            item={k:p[k] for k in ('population','agriculture','tools','food','resources','buildings','points','one_use','deferred','unplaced')}
            item['card_count']=len(p['cards'])
            item['cards']=p['cards'] if i==player or s['terminated'] else None
            players.append(item)
        return dict(players=players,phase=s['phase'],round=s['round'],first_player=s['first_player'],
                    placements=s['placements'],market=s['market'],
                    buildings=[stack[-1] if stack else None for stack in s['building_stacks']],
                    building_stack_sizes=[len(stack) for stack in s['building_stacks']],
                    card_deck_size=len(s['card_deck']),pending=s['pending'],
                    public_history=s['public_history'],own_projected_score=final_breakdown(s['players'][player])['total'])

    def validate(self):
        super().validate();s=self.state
        cards=s['card_deck']+[c for c in s['market'] if c is not None]
        buildings=s['removed_buildings']+sum(s['building_stacks'],[])
        assert len(s['market'])==4 and len(s['building_stacks'])==2 and len(s['removed_buildings'])==14
        assert set(s['placements'])==set(LOCATIONS)
        for p,player in enumerate(s['players']):
            cards+=player['cards'];buildings+=player['buildings']
            assert 5<=player['population']<=10 and 0<=player['agriculture']<=10
            assert 0<=player['unplaced']<=player['population']
            assert player['food']>=0 and all(type(n) is int and n>=0 for n in player['resources'])
            assert len(player['resources'])==4 and len(player['tools'])<=3
            assert all(type(t['value']) is int and 1<=t['value']<=4 and type(t['used']) is bool for t in player['tools'])
            values=[t['value'] for t in player['tools']]
            assert not values or max(values)-min(values)<=1
            assert len(values)==3 or all(v==1 for v in values)
            assert len(set(player['one_use']))==len(player['one_use']) and len(set(player['deferred']))==len(player['deferred'])
            assert all(cid in player['cards'] and CARDS[cid]['effect']['type']=='one_use_tool' for cid in player['one_use'])
            assert all(cid in player['cards'] and CARDS[cid]['effect']['type']=='deferred_resources' for cid in player['deferred'])
            placed=sum(v[p] for v in s['placements'].values())
            assert placed+player['unplaced']<=player['population']
        assert sorted(cards)==list(CARDS) and sorted(buildings)==list(BUILDINGS)
        for loc,occ in s['placements'].items():
            assert len(occ)==2 and all(type(n) is int and n>=0 for n in occ)
            if loc!='hunt': assert sum(n>0 for n in occ)<=1
            if loc=='hut': assert all(n in (0,2) for n in occ)
            elif loc in GATHER[1:]: assert sum(occ)<=7
            elif loc!='hunt': assert sum(occ)<=1
        assert sum(any(s['placements'][loc]) for loc in VILLAGE)<=2
        held=[sum(p['food'] for p in s['players'])]+[sum(p['resources'][i] for p in s['players']) for i in range(4)]
        assert held==[a-b for a,b in zip(s['produced'],s['spent'])]
        phase=s['interrupt']['phase'] if s['phase']=='resource_window' else s['phase']
        if phase=='place':
            assert not s['effects'] and s['pending'] is None
            for p,player in enumerate(s['players']):assert sum(v[p] for v in s['placements'].values())+player['unplaced']==player['population']
        if phase in ('roll','upgrade','draft_die'):assert s['effects']
        if s['terminated']:
            assert s['phase']=='finished' and s['feeding_index']==2
            assert s['final_scores']==[final_breakdown(p)['total'] for p in s['players']]

    @staticmethod
    def features(observation,action):
        p=observation['current_player'];player=observation['players'][p];typ=action['type'];f=[0.]*24
        remaining=max(observation['building_stack_sizes'])
        horizon=max(1.,min(8.,remaining*1.2,observation['card_deck_size']/3+1))
        deficit=max(0,player['population']-player['agriculture']-player['food'])
        culture,mult=card_totals(player['cards'])
        f[21]=player['agriculture']/10;f[22]=player['population']/10;f[23]=horizon/8
        if typ=='take_two':
            f[0]=2.5*sum(n*v for n,v in zip(action['resources'],VALUES))/4
            for i,n in enumerate(action['resources']):f[12+i]=n
        elif typ=='wait':f[0]=0
        elif typ=='place':
            loc=action['location'];n=action['count'];f[1]=n
            if loc=='tool':
                total=sum(t['value'] for t in player['tools']);f[2]=max(0,12-total)>0
                f[0]=(2.8+.35*horizon+mult['toolmakers'])*f[2]
            elif loc=='field':
                f[3]=int(player['agriculture']<10);f[0]=(3.2+.5*horizon+mult['farmers'])*f[3]
                if player['agriculture']>=player['population']+1:f[0]*=.25
            elif loc=='hut':
                f[4]=int(player['population']<10);f[0]=(1.4+.65*horizon+mult['shamans'])*f[4]
            elif loc in GATHER:
                divisor=2 if loc=='hunt' else VALUES[RESOURCES.index(loc)]
                expected=max(0,(3.5*n-(divisor-1)/2)/divisor);f[5]=expected
                if loc=='hunt':
                    target=max(0,2*(player['population']-player['agriculture'])-player['food'])
                    f[6]=min(expected,target);f[0]=(2*f[6]+.1*expected)/(n**.75)
                    if deficit>0:f[0]+=min(expected,deficit)/n
                else:
                    i=RESOURCES.index(loc);f[12+i]=expected
                    reserve=player['resources'][i]
                    needed=max([0]+[max(0,BUILDINGS[b].get('cost',[0]*4)[i]-reserve) for b in observation['buildings'] if b is not None])
                    f[7]=min(expected,needed)
                    f[0]=(2.6*expected+.9*f[7])/(n**.8) * (1 if reserve<6 else .4)
                    if loc=='wood':f[0]+=.6
            elif loc.startswith('card'):
                i=int(loc[4:]);cid=observation['market'][i];f[8]=public_card_value(player,cid)
                available=sum(player['resources'])+sum(observation['placements'][r][p]*3.5/VALUES[j] for j,r in enumerate(RESOURCES))
                f[9]=max(0,i+1-available)
                f[0]=f[8]/(1+.55*(i+1))-4*f[9]
            else:
                i=int(loc[8:]);b=BUILDINGS[observation['buildings'][i]];opts=building_payments(player['resources'],b)
                f[10]=max([0]+[sum(n*v for n,v in zip(c,VALUES)) for c in opts]);f[0]=(.4*f[10]+mult['builders']) if opts else -1
        elif typ=='resolve':
            loc=action['location'];f[0]=10 if loc=='tool' else 5 if loc in GATHER else 1
            if loc.startswith('card'):
                i=int(loc[4:]);f[0]=7 if CARDS[observation['market'][i]]['effect']['type'] in ('tool','agriculture','deferred_resources') else 2
                if sum(player['resources'])<i+1:f[0]=-3
            if loc.startswith('building'):
                b=BUILDINGS[observation['buildings'][int(loc[8:])]]
                f[0]=2 if building_payments(player['resources'],b) else -3
        elif typ=='decline':f[0]=-1
        elif typ in ('buy_card','build'):
            cost=action['resources'];f[11]=sum(n*v for n,v in zip(cost,VALUES))
            if typ=='buy_card':f[8]=public_card_value(player,observation['pending']['card']);f[0]=f[8]-.45*f[11]
            else:f[10]=f[11];f[0]=.55*f[11]+mult['builders']
            for i,n in enumerate(cost):f[12+i]=-n
        elif typ=='use_tools':
            bonus=sum(player['tools'][i]['value'] for i in action['permanent'])+sum(CARDS[c]['effect']['amount'] for c in action['one_use'])
            res=observation['pending']['resource'];divisor=2 if res=='food' else VALUES[RESOURCES.index(res)]
            base=sum(observation['pending']['dice']);f[16]=(base+bonus)//divisor-base//divisor;f[17]=bonus
            future=sum(observation['placements'][r][p]>0 for r in GATHER)
            f[0]=f[16]*(2.5 if res=='food' and deficit else 2)-(.12 if future else 0)*bonus-.25*len(action['one_use'])
        elif typ=='upgrade_tool':
            f[0]=1-int(player['tools'][action['index']]['used'])
        elif typ=='choose_die':
            d=action['value'];f[0]=[0,2.6,2.8,3.2,3.6,3.2+.2*horizon,3.8+.3*horizon][d]
            if d==5 and sum(t['value'] for t in player['tools'])==12:f[0]=0
            if d==6 and player['agriculture']==10:f[0]=0
        elif typ=='starve':f[0]=-10
        elif typ=='feed':
            f[11]=sum(n*v for n,v in zip(action['resources'],VALUES));f[0]=-.6*f[11]
        return f


GAME_CLASS=StoneAge
