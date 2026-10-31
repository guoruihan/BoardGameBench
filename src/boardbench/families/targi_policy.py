"""Opt-in public-history guard for Targi bots; this does not alter game legality.

A Trader activation only changes the acting player's resource vector. Returning
that vector to an earlier one makes no game progress. The original rules permit
such exchanges; this wrapper chooses not to take them. It reconstructs history
from the observation, so its extra behavior needs no privileged snapshot/state.
"""

GUARD_ID = 'targi_trader_acyclic_v1'


def trader_acyclic_actions(observation, actions):
    """Keep candidate actions that do not revisit this Trader's resource states.

    Only public own resources and the uninterrupted suffix of actual public
    trades are read. An environment's legal-action list is never mutated.
    """
    if observation.get('game_id') != 'targi':
        raise ValueError('Targi Trader guard requires a Targi observation')
    if observation.get('phase') != 'trader':
        return list(actions)
    player = observation['player']
    resources = tuple(observation['players'][player]['resources'])
    previous = list(resources)
    visited = {resources}
    for event in reversed(observation['history']):
        action = event['action']
        if (event['player'] != player or event['round'] != observation['round']
                or action['type'] != 'trade'):
            break
        # Undo the observed exchange; Trader has no other automatic side effects.
        previous[action['give']] += action['rate']
        previous[action['take']] -= 1
        visited.add(tuple(previous))
    candidates = []
    for action in actions:
        if action['type'] != 'trade':
            candidates.append(action)
            continue
        after = list(resources)
        after[action['give']] -= action['rate']
        after[action['take']] += 1
        if tuple(after) not in visited:
            candidates.append(action)
    if not candidates:
        raise ValueError('Trader guard expected a legal done action')
    return candidates


class TargiTraderAcyclicPolicy:
    """Wrap a public-observation policy under an explicit new policy identity.

    The caller should record ``decision_guard: targi_trader_acyclic_v1`` in its
    policy artifact and evaluate it afresh. Existing unguarded artifacts retain
    their old semantics. Random, linear and neural scorers can all be wrapped.
    """
    guard_id = GUARD_ID

    def __init__(self, policy):
        self.policy = policy

    def decide(self, observation, actions):
        return self.policy.decide(observation, trader_acyclic_actions(observation, actions))
