"""Exact replacement for repeated Counter-based geometric goal enumeration.

Only static stack transitions are compiled. All state and supply checks remain
public and identical to v123.card_goals; this does not strengthen game rules.
"""
from itertools import product

from boardbench.environments.harmonies import CARDS, COLORS, PATTERNS, STACKS
from boardbench.v123.policies import available_tokens

_EXTENSIONS = {}
for sid, current in enumerate(STACKS):
    for target, desired in enumerate(STACKS):
        if desired[:len(current)] == current:
            missing = desired[len(current):]
            _EXTENSIONS[sid,target] = (len(missing), tuple((i,missing.count(c)) for i,c in enumerate(COLORS) if c in missing))


def card_goals_compiled(obs, card_id, placed_count=0):
    available = available_tokens(obs)
    supply = tuple(available[c] for c in COLORS)
    board, goals = obs['board'], []
    track = CARDS[card_id]['score_by_placed_count']
    marginal = track[placed_count+1]-track[placed_count]
    for anchor, orientations in enumerate(PATTERNS[card_id]):
        if board[anchor]['animal'] is not None:
            continue
        for orientation, pattern in orientations.items():
            options = []
            for cell, allowed in pattern:
                current = board[cell]
                candidates = []
                for target in allowed:
                    extension = _EXTENSIONS.get((current['stack_id'],target))
                    if extension is not None and not (extension[0] and current['animal'] is not None):
                        candidates.append((cell,target,*extension))
                if not candidates:
                    break
                options.append(candidates)
            else:
                best = None
                for choice in product(*options):
                    need = [0]*len(COLORS)
                    count = 0
                    for cell,target,missing,requirements in choice:
                        count += missing
                        for color,n in requirements:
                            need[color] += n
                    if any(n>limit for n,limit in zip(need,supply)):
                        continue
                    key = (count,tuple(option[1] for option in choice))
                    if best is None or key<best[0]:
                        best=(key,choice,need)
                if best is not None:
                    key,choice,need=best
                    goals.append({'anchor':anchor,'targets':{c:t for c,t,_,_ in choice},
                        'need':{c:n for c,n in zip(COLORS,need) if n},'missing':key[0],
                        'card_id':card_id,'orientation':orientation,'value':marginal/(1+key[0])})
    return sorted(goals,key=lambda g:-g['value'])
