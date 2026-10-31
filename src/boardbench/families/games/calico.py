"""Calico standard duo: authentic teal/purple boards and chosen size cats.

The rulebook permits choosing the three cats. This configuration fixes Millie,
Tibbit, Coconut, retains random pattern pairs and all standard goal choices.
"""
from collections import Counter
from functools import lru_cache
from itertools import permutations
import json
from math import factorial
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/calico.json').read_text())
DIRECTIONS = ((1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1))
PLAY_COORDS = tuple(map(tuple, DATA['playable_cells']))
GOAL_COORDS = tuple(map(tuple, DATA['goal_slots']))
BORDER_COORDS = tuple((c['q'], c['r']) for c in DATA['boards'][0]['printed_cells'])
COORDS = PLAY_COORDS + GOAL_COORDS + BORDER_COORDS
INDEX = {coord:i for i, coord in enumerate(COORDS)}
ADJ = tuple(tuple(INDEX[(q+dq,r+dr)] for dq,dr in DIRECTIONS if (q+dq,r+dr) in INDEX)
            for q,r in COORDS)
GOAL_INDICES = (22, 23, 24)
GOALS = {g['id']:g for g in DATA['goals']}
BORDERS = tuple(tuple(c['color']*6+c['pattern'] for c in b['printed_cells']) for b in DATA['boards'])
CATS = tuple(DATA['cats'])
TARGET_PERMS = {g:tuple(set(permutations(tuple(v['counts'])+(0,)*(6-len(v['counts'])))))
                for g,v in GOALS.items()}


def component(board, cell, attribute):
    if board[cell] < 0:
        return set()
    val = board[cell]//6 if attribute == 'color' else board[cell]%6
    found, todo = {cell}, [cell]
    while todo:
        c = todo.pop()
        for n in ADJ[c]:
            if n not in found and board[n] >= 0 and (board[n]//6 if attribute == 'color' else board[n]%6) == val:
                found.add(n); todo.append(n)
    return found


def goal_score(board, cell, goal):
    if goal is None or any(board[n] < 0 for n in ADJ[cell]):
        return 0
    target = sorted(GOALS[goal]['counts'])
    matches = sum(sorted(Counter(board[n]//6 if a==0 else board[n]%6 for n in ADJ[cell]).values()) == target
                  for a in (0,1))
    return 0 if matches == 0 else GOALS[goal]['single' if matches==1 else 'double']


@lru_cache(maxsize=4096)
def completion_probability(counts, goal):
    known = counts+(0,)*(6-len(counts))
    missing = 6-sum(counts)
    if missing < 0:
        return 0.
    ways = 0
    for target in TARGET_PERMS[goal]:
        if all(a <= b for a,b in zip(known,target)):
            ways += factorial(missing)//_product_factorials(b-a for a,b in zip(known,target))
    return ways/(6**missing)


def _product_factorials(values):
    out = 1
    for v in values:
        out *= factorial(v)
    return out


def goal_expectation(board, cell, goal):
    if goal is None:
        return 0.
    probs=[]
    for attr in (0,1):
        counts=Counter((board[n]//6 if attr==0 else board[n]%6) for n in ADJ[cell] if board[n]>=0)
        probs.append(completion_probability(tuple(sorted(counts.values(),reverse=True)),goal))
    pc,pp=probs; spec=GOALS[goal]
    return spec['single']*(pc+pp)+(spec['double']-2*spec['single'])*pc*pp


def placement_signals(board, goals, buttons, cats, pattern_cats, code, cell):
    """Pure public one-placement evaluation; reward history remains immutable."""
    new = list(board);new[cell]=code
    color,pattern=code//6,code%6
    ci=pattern_cats[pattern]; cat=CATS[ci]
    signals=[]
    for attr,anchors,threshold,points in (
            ('color', {b['cell'] for b in buttons},3,3),
            ('pattern',{c['cell'] for c in cats},cat['size'],cat['points'])):
        merged=component(new,cell,attr)
        awarded=bool(merged & anchors)
        gain=points if len(merged)>=threshold and not awarded else 0
        oldparts=set()
        oldpotential=0.
        for n in ADJ[cell]:
            if n in merged and n not in oldparts:
                part=component(board,n,attr);oldparts.update(part)
                if not part & anchors:
                    oldpotential+=points*(min(len(part),threshold)/threshold)**2
        newpotential=0 if awarded or gain else points*(min(len(merged),threshold)/threshold)**2
        signals.extend((gain,newpotential-oldpotential,len(merged),int(awarded)))
    color_gain,color_progress,color_size,color_used,cat_gain,cat_progress,cat_size,cat_used=signals
    rainbow=int(color_gain>0 and color not in {b['color'] for b in buttons}
                and len({b['color'] for b in buttons})==5)*3
    goal_delta=sum(goal_expectation(new,g,goal)-goal_expectation(board,g,goal)
                   for g,goal in zip(GOAL_INDICES,goals) if cell in ADJ[g])
    score=color_gain+cat_gain+rainbow+1.5*goal_delta+color_progress+cat_progress
    return [score,color_gain,cat_gain,rainbow,goal_delta,color_progress,cat_progress,
            color_size,cat_size,color_used,cat_used,
            sum(board[n]>=0 for n in ADJ[cell]),sum(board[n]==-1 for n in ADJ[cell]),
            sum(cell in ADJ[g] for g in GOAL_INDICES)]


class Calico(DuoGame):
    game_id='calico'
    config_id='standard_teal_purple_sizecats_duo'

    def _reset(self,first_player):
        patterns=list(range(6));self.rng.shuffle(patterns)
        assignments=[0]*6
        for ci in range(3):
            for pat in patterns[2*ci:2*ci+2]:assignments[pat]=ci
        offers=[]
        for p in (0,1):
            choices=list(GOALS);self.rng.shuffle(choices);offers.append(choices[:4])
        deck=list(range(108));self.rng.shuffle(deck)
        self.state.update(boards=[[None]*22 for _ in (0,1)], hands=[[],[]],
                          deck=deck,market=[],goal_offers=offers,goals=[None,None],
                          pattern_cats=assignments,buttons=[[],[]],cats=[[],[]],
                          rainbow=[False,False],turns=[0,0],phase='goals',public_history=[])

    def _public_board(self,player):
        return [uid//3 if uid is not None else -1 for uid in self.state['boards'][player]]+[-2]*3+list(BORDERS[player])

    def _observe(self,player):
        s=self.state
        return dict(phase=s['phase'],boards=[self._public_board(p) for p in (0,1)],
                    coordinates=[list(c) for c in COORDS],playable_indices=list(range(22)),
                    goal_indices=list(GOAL_INDICES),board_ids=['teal','purple'],
                    hand=[uid//3 for uid in s['hands'][player]],
                    hand_sizes=[len(h) for h in s['hands']], market=[u//3 for u in s['market']],
                    deck_size=len(s['deck']),goal_offers=s['goal_offers'],goals=s['goals'],
                    pattern_cats=s['pattern_cats'],cat_rules=DATA['cats'],goal_rules=DATA['goals'],
                    buttons=s['buttons'],cats=s['cats'],rainbow=s['rainbow'],turns=s['turns'],
                    public_history=s['public_history'],score_breakdown=self.score_breakdown())

    def _legal_actions(self):
        s,p=self.state,self.state['current_player']
        if s['phase']=='goals':
            return [dict(type='choose_goals',goals=list(gs)) for gs in permutations(s['goal_offers'][p],3)]
        if s['phase']=='draft':
            return [dict(type='draft',slot=i) for i in range(len(s['market']))]
        return [dict(type='place',hand=i,cell=c) for i in range(len(s['hands'][p]))
                for c in range(22) if s['boards'][p][c] is None]

    def _step(self,action):
        s,p=self.state,self.state['current_player']
        if action['type']=='choose_goals':
            s['goals'][p]=list(action['goals'])
            s['public_history'].append(dict(player=p,**action))
            if s['goals'][1-p] is None:
                s['current_player']=1-p
            else:
                for seat in (0,1):s['hands'][seat]=[s['deck'].pop(),s['deck'].pop()]
                s['market']=[s['deck'].pop() for _ in range(3)]
                s['current_player']=s['first_player'];s['phase']='place'
        elif action['type']=='place':
            uid=s['hands'][p].pop(action['hand']);cell=action['cell']
            s['boards'][p][cell]=uid;s['turns'][p]+=1
            s['public_history'].append(dict(type='place',player=p,cell=cell,tile=uid//3))
            self._award(p,cell)
            s['phase']='draft'
        else:
            slot=action['slot'];uid=s['market'][slot]
            s['hands'][p].append(uid)
            s['public_history'].append(dict(type='draft',player=p,slot=slot,tile=uid//3))
            s['market'][slot]=s['deck'].pop()
            if s['turns']==[22,22]:
                scores=self.scores()
                keys=[(scores[q],len(s['cats'][q]),len(s['buttons'][q])+int(s['rainbow'][q])) for q in (0,1)]
                self.finish([q for q in (0,1) if keys[q]==max(keys)],reason='both_quilts_filled')
            else:
                s['current_player']=1-p;s['phase']='place'

    def _award(self,p,cell):
        s=self.state;board=self._public_board(p);color=board[cell]//6;pattern=board[cell]%6
        colors=component(board,cell,'color')
        if len(colors)>=3 and not colors & {b['cell'] for b in s['buttons'][p]}:
            s['buttons'][p].append(dict(color=color,cell=cell))
            if len({b['color'] for b in s['buttons'][p]})==6:s['rainbow'][p]=True
        ci=s['pattern_cats'][pattern]
        patterns=component(board,cell,'pattern')
        if len(patterns)>=CATS[ci]['size'] and not patterns & {c['cell'] for c in s['cats'][p]}:
            s['cats'][p].append(dict(cat=ci,pattern=pattern,cell=cell))

    def score_breakdown(self):
        result=[]
        for p in (0,1):
            s=self.state;board=self._public_board(p)
            goals=[goal_score(board,c,g) for c,g in zip(GOAL_INDICES,s['goals'][p] or [None]*3)]
            cats=sum(CATS[c['cat']]['points'] for c in s['cats'][p])
            buttons=3*(len(s['buttons'][p])+int(s['rainbow'][p]))
            result.append(dict(goals=goals,cats=cats,buttons=buttons,total=sum(goals)+cats+buttons))
        return result

    def scores(self):return [v['total'] for v in self.score_breakdown()]

    def validate(self):
        super().validate();s=self.state
        tiles=list(s['deck'])+s['market']
        if s['phase'] not in ('goals','place','draft'):raise ValueError('invalid phase')
        if sorted(s['pattern_cats'])!=[0,0,1,1,2,2]:raise ValueError('invalid cat pattern assignment')
        for p in (0,1):
            if len(s['boards'][p])!=22:raise ValueError('invalid board size')
            tiles+=s['hands'][p]+[u for u in s['boards'][p] if u is not None]
            if s['turns'][p]!=sum(u is not None for u in s['boards'][p]):raise ValueError('placement count mismatch')
            if len(s['goal_offers'][p])!=4 or len(set(s['goal_offers'][p]))!=4 or not set(s['goal_offers'][p])<=set(GOALS):raise ValueError('invalid goals offered')
            if s['goals'][p] is not None and (len(s['goals'][p])!=3 or len(set(s['goals'][p]))!=3 or not set(s['goals'][p])<=set(s['goal_offers'][p])):raise ValueError('invalid chosen goals')
            board=self._public_board(p)
            for b in s['buttons'][p]:
                if not 0<=b['cell']<22 or board[b['cell']]//6!=b['color'] or len(component(board,b['cell'],'color'))<3:raise ValueError('invalid button award')
            for c in s['cats'][p]:
                if not 0<=c['cell']<22 or board[c['cell']]%6!=c['pattern'] or s['pattern_cats'][c['pattern']]!=c['cat'] or len(component(board,c['cell'],'pattern'))<CATS[c['cat']]['size']:raise ValueError('invalid cat award')
            if s['rainbow'][p]!=(len({b['color'] for b in s['buttons'][p]})==6):raise ValueError('invalid rainbow')
            if len({b['cell'] for b in s['buttons'][p]})!=len(s['buttons'][p]) or len({c['cell'] for c in s['cats'][p]})!=len(s['cats'][p]):raise ValueError('duplicate award')
        if sorted(tiles)!=list(range(108)):raise ValueError('tile conservation failure')
        if s['phase']=='goals':
            if s['market'] or any(s['hands']):raise ValueError('premature deal')
        else:
            expected=[2,2]
            if s['phase']=='draft' and not s['terminated']:expected[s['current_player']]=1
            if [len(h) for h in s['hands']]!=expected or len(s['market'])!=3:raise ValueError('hand/market phase mismatch')
        if s['terminated'] and s['turns']!=[22,22]:raise ValueError('premature ending')

    @staticmethod
    def features(observation,action):
        p=observation['current_player'];s=observation;board=s['boards'][p]
        phase=action['type'];code=0
        if phase=='choose_goals':
            value=sum(goal_expectation(board,c,g) for c,g in zip(GOAL_INDICES,action['goals']))
            signals=[value,0,0,0,value]+[0]*9
        else:
            if phase=='place':
                code=s['hand'][action['hand']]
                signals=placement_signals(board,s['goals'][p],s['buttons'][p],s['cats'][p],s['pattern_cats'],code,action['cell'])
            else:
                code=s['market'][action['slot']]
                choices=[placement_signals(board,s['goals'][p],s['buttons'][p],s['cats'][p],s['pattern_cats'],code,c)
                         for c in range(22) if board[c]<0]
                signals=max(choices,key=lambda v:v[0]) if choices else [0]*14
        values=signals+[code//6,code%6,s['turns'][p]/22,22-s['turns'][p],
                        s['score_breakdown'][p]['total'],s['score_breakdown'][1-p]['total'],
                        float(phase=='choose_goals'),float(phase=='place'),float(phase=='draft'),
                        s['deck_size']/108]
        return [float(v) for v in values]


GAME_CLASS=Calico
