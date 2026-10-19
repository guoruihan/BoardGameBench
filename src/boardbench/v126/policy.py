"""Small CPU students; exported NumPy weights avoid a Torch runtime per rollout."""
from collections import OrderedDict
import hashlib
import json

import numpy as np

from boardbench.artifacts.store import digest
from boardbench.v12.encoding import encode, action_map
from boardbench.v125.policies import action_key
from boardbench.v125.search import Search


class Student:
    def __init__(self, checkpoint, sha256):
        if digest(checkpoint) != sha256:
            raise ValueError('student checkpoint hash mismatch')
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.weights = {k: archive[k].copy() for k in archive.files}
        expected = {'body.0.weight', 'body.0.bias', 'body.2.weight', 'body.2.bias',
                    'actor.weight', 'actor.bias', 'critic.weight', 'critic.bias'}
        if set(self.weights) != expected or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError('invalid student arrays')
        h = self.weights['body.0.bias'].shape[0]
        shapes = {'body.0.weight': (h,1346), 'body.0.bias': (h,), 'body.2.weight': (h,h),
                  'body.2.bias': (h,), 'actor.weight': (240,h), 'actor.bias': (240,),
                  'critic.weight': (1,h), 'critic.bias': (1,)}
        if any(self.weights[k].shape != shape for k,shape in shapes.items()):
            raise ValueError('student shape mismatch')
        self.calls = self.hits = 0
        self.cache = OrderedDict()

    def logits(self, obs):
        data = encode(obs, allow_stop=False)
        x = np.asarray(data['features'], dtype=np.float32)
        w = self.weights
        h = np.tanh(w['body.0.weight'] @ x + w['body.0.bias'])
        h = np.tanh(w['body.2.weight'] @ h + w['body.2.bias'])
        logits = w['actor.weight'] @ h + w['actor.bias']
        if not np.isfinite(logits).all():
            raise ValueError('nonfinite student inference')
        return np.where(data['action_mask'], logits, -np.inf)

    def rank(self, obs):
        mapping = action_map(obs, 'slots240_v1', allow_stop=False)
        logits = self.logits(obs)
        return [(float(logits[i]), mapping[i]) for i in sorted(mapping, key=lambda i: (-logits[i], i))]

    def decide(self, obs):
        self.calls += 1
        key = hashlib.sha256(json.dumps(obs,sort_keys=True,separators=(',',':')).encode()).digest()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        result = self.rank(obs)[0][1]
        self.cache[key] = dict(result)
        if len(self.cache) > 32768:
            self.cache.popitem(last=False)
        return dict(result)


class LearnedSearch(Search):
    def __init__(self, seed, variant, seconds, checkpoint, checkpoint_sha256, **kwargs):
        if variant not in ('student_roots', 'student_rollout', 'student', 'dagger'):
            raise ValueError('unknown learned search')
        super().__init__(seed, 'mixed_roots', seconds, **kwargs)
        self.student = Student(checkpoint, checkpoint_sha256)
        self.learned_variant = variant
        if variant == 'student_rollout':
            self.policy = self.student

    def rank_candidates(self, observation):
        original = super().rank_candidates(observation)
        if self.learned_variant != 'student_roots' or len(original) < 6:
            return original
        seen = {action_key(a) for _, a in original}
        proposal = next((p for p in self.student.rank(observation) if action_key(p[1]) not in seen), None)
        # Fixed six candidates, preserve root0 and first five old candidates.
        return original[:5] + [proposal] if proposal is not None else original

    def _decide(self, observation, context, begin):
        if self.learned_variant == 'student':
            action = self.student.decide(observation)
            self.last_search = {'candidate_count': 1, 'simulation_steps': 0,
                                'learned_variant': 'student', 'candidates': []}
            return action
        expert = super()._decide(observation, context, begin)
        self.last_search['learned_variant'] = self.learned_variant
        if self.learned_variant == 'dagger':
            self.last_search['expert_action'] = expert
            # Pure student roll-in; expensive expert query only supplies a label.
            return self.student.decide(observation)
        return expert
