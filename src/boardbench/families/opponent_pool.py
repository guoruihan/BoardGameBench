"""Frozen historical opponents, decayed training statistics and explicit selection.

Evaluation results never enter the training sampler's win statistics. All state
is JSON serializable and archived members are never deleted by active selection.
"""
from copy import deepcopy
import numpy as np


def sampling_probabilities(ids, statistics):
    if not ids:
        raise ValueError('An opponent pool cannot be empty')
    weights = []
    for key in ids:
        s = statistics.get(key, {})
        p = (s.get('wins', 0.) + .5*s.get('draws', 0.) + 1.)/(s.get('games', 0.)+2.)
        if not 0 <= p <= 1:
            raise ValueError('Invalid opponent statistics')
        weights.append(p*(1-p))
    w = np.asarray(weights, dtype=float)
    return .5/len(ids) + .5*w/w.sum() if w.sum() > 0 else np.full(len(ids), 1/len(ids))


def behavior_distance(a, b):
    if not a or not b or len(a) != len(b):
        return 0.
    return sum(x != y for x, y in zip(a, b))/len(a)


class OpponentPool:
    def __init__(self, initial, capacity=12, decay=.95):
        if capacity < 3 or not 0 < decay <= 1:
            raise ValueError('Invalid pool capacity/decay')
        key = initial['id']
        self.state = dict(capacity=capacity, decay=decay, members={key:deepcopy(initial)},
                          initial=key, latest=key, best=key, active=[key], statistics={},
                          audit_cursor=0, events=[])

    @classmethod
    def restore(cls, state):
        obj = cls.__new__(cls)
        obj.state = deepcopy(state)
        if not obj.state['active'] or not set(obj.state['active']) <= set(obj.state['members']):
            raise ValueError('Corrupt active opponent pool')
        return obj

    def distribution(self):
        ids = list(self.state['active'])
        return ids, sampling_probabilities(ids, self.state['statistics'])

    def update_training(self, results):
        # One decay per complete sampling block, never per action or evaluation.
        stats = self.state['statistics']
        for values in stats.values():
            for k in ('wins', 'draws', 'games'):
                values[k] *= self.state['decay']
        for opponent_id, reward in results:
            if opponent_id not in self.state['members']:
                raise ValueError('Unknown frozen opponent')
            if reward not in (-1., 0., 1.):
                raise ValueError('Only completed-game outcomes update opponent statistics')
            values = stats.setdefault(opponent_id, dict(wins=0., draws=0., games=0.))
            values['games'] += 1
            values['wins'] += reward == 1
            values['draws'] += reward == 0

    def add(self, member, confirmed_best=False):
        key = member['id']
        if key in self.state['members']:
            raise ValueError('Archive identities are immutable')
        self.state['members'][key] = deepcopy(member)
        self.state['latest'] = key
        if confirmed_best:
            self.state['best'] = key

    def select(self):
        s, members = self.state, self.state['members']
        active, roles = [], {}
        def retain(key, role):
            if key not in active and len(active) < s['capacity']:
                active.append(key)
            if key in active:
                roles.setdefault(key, []).append(role)
        for role in ('initial', 'latest', 'best'):
            retain(s[role], role)
        counters = sorted((k for k, v in members.items() if v.get('confirmed_counter')),
                          key=lambda k:members[k].get('counter_upper', 1.))[:2]
        for key in counters:
            retain(key, 'confirmed_counter')
        scores = [m['quality'] for m in members.values() if m.get('quality') is not None]
        best_quality = max(scores) if scores else None
        candidates = [k for k, v in members.items() if k not in active and
                      best_quality is not None and v.get('quality') is not None and
                      v['quality'] >= best_quality-.10 and v.get('fingerprint')]
        while candidates and len(active) < s['capacity']:
            def rank(key):
                distance = min(behavior_distance(members[key].get('fingerprint'),
                                                members[x].get('fingerprint')) for x in active)
                return distance, members[key]['quality']
            key = max(candidates, key=rank)
            distance, _ = rank(key)
            candidates.remove(key)
            # Differences below 1% with equivalent panel vectors do not fill slots.
            if distance < .01 and any(max((abs(a-b) for a, b in zip(
                    members[key].get('panel_vector', []), members[x].get('panel_vector', []))),
                    default=0.) < .05 for x in active):
                continue
            retain(key, 'quality_diversity')
        previous = list(s['active'])
        s['active'] = active
        s['events'].append(dict(active=active, roles=roles, removed=[x for x in previous if x not in active],
                                reason='validation_boundary_quality_and_behavior'))
        return active

    def audit_members(self, count=2):
        inactive = [k for k in self.state['members'] if k not in self.state['active']]
        if not inactive:
            return []
        start = self.state['audit_cursor'] % len(inactive)
        chosen = [inactive[(start+i) % len(inactive)] for i in range(min(count, len(inactive)))]
        self.state['audit_cursor'] += len(chosen)
        return chosen


def pool_seed(split, run_seed, iteration=0, index=0):
    """Exact 53-bit integers; new v170 partitions cannot overlap V160 seeds.

    Arm is intentionally absent: paired A/B share environment seeds. Independent
    training replicate IDs occupy different bits. Validation/fingerprint seeds
    use a fixed replicate ID and never depend on training progress.
    """
    partitions = {'train':8, 'quick':9, 'confirm':10, 'fingerprint':11, 'audit':12, 'smoke':13}
    if split not in partitions or any(type(v) is not int for v in (run_seed, iteration, index)):
        raise ValueError('Invalid seed namespace')
    if not (0 <= run_seed < 2**12 and 0 <= iteration < 2**24 and 0 <= index < 2**12):
        raise ValueError('Seed fields overflow')
    return (partitions[split] << 48) | (run_seed << 36) | (iteration << 12) | index
