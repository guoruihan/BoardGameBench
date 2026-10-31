"""Optional public-state repetition handling for Century: A New World bots.

This changes only policy action selection, never legal moves or game endings.
A fresh wrapper belongs to one player in one episode. Reusing it across episodes
requires reset(); restoring an environment alone does not restore this memory.
"""
from collections import defaultdict
import hashlib
import json


GUARD_ID = 'century_new_world_public_repeat_v1'
GAME_ID = 'century_a_new_world'


def _action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'), allow_nan=False)


def public_turn_fingerprint(observation):
    """Hash the supplied public view, removing only history and pure clocks.

    Keep market, deck top/count, stack tops/counts, covers, occupation, acquired
    components, resources, player identity and final-round status. No environment
    reference or hidden snapshot is accepted. Avoid copying unbounded history.
    """
    if observation.get('game_id') != GAME_ID:
        raise ValueError('New World repetition guard requires a New World observation')
    view = {k: v for k, v in observation.items() if k not in ('history', 'decision_index')}
    view['players'] = [{k: v for k, v in p.items() if k != 'turns'}
                       for p in observation['players']]
    encoded = json.dumps(view, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class CenturyNewWorldRepeatPolicy:
    """On revisiting a turn state, let the wrapped policy choose least-tried moves.

    First visits preserve the original candidate sequence and policy randomness.
    Only phase=turn is tracked; discard/explore choices are untouched. Candidate
    normalization or contextual scoring can change when the subset changes, so
    this is not necessarily the original full-set ranking with one move removed.
    Least-tried selection is a heuristic, not a guarantee of eventual termination.
    """
    guard_id = GUARD_ID

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        """Clear guard memory; the caller must also reset the underlying policy."""
        self.action_counts = defaultdict(dict)
        self.restricted_decisions = 0

    def decide(self, observation, actions):
        if observation.get('game_id') != GAME_ID:
            raise ValueError('New World repetition guard requires a New World observation')
        if observation.get('phase') != 'turn':
            return self.policy.decide(observation, actions)
        if not actions:
            raise ValueError('New World repetition guard requires legal actions')
        fingerprint = public_turn_fingerprint(observation)
        counts = self.action_counts[fingerprint]
        keys = [_action_key(a) for a in actions]
        smallest = min(counts.get(k, 0) for k in keys)
        candidates = [a for a, k in zip(actions, keys) if counts.get(k, 0) == smallest]
        restricted = len(candidates) != len(actions)
        action = self.policy.decide(observation, candidates if restricted else actions)
        if action not in candidates:
            raise ValueError('Underlying policy returned an action outside the guard candidates')
        chosen = _action_key(action)
        counts[chosen] = counts.get(chosen, 0) + 1
        self.restricted_decisions += int(restricted)
        return action
