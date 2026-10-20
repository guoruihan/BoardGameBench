"""Search labels student roll-ins without ever controlling the real trajectory."""
import argparse
from collections import OrderedDict
import json

import numpy as np

from boardbench.artifacts.store import digest, read_json
from boardbench.v12.encoding import encode, action_map
from boardbench.v124.episode import run_episode as base_episode
from boardbench.v125.policies import action_key, mixed_candidates
from boardbench.v125.search import Search
from boardbench.v126.policy import Student


class Student64(Student):
    def __init__(self, checkpoint, sha256, *, stochastic=False, seed=0):
        if digest(checkpoint) != sha256:
            raise ValueError('roll-in checkpoint hash mismatch')
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.weights = {k: archive[k].copy() for k in archive.files}
        h = self.weights['body.0.bias'].shape[0]
        expected = {'body.0.weight': (h,1346), 'body.0.bias': (h,), 'body.2.weight': (h,h),
                    'body.2.bias': (h,), 'actor.weight': (240,h), 'actor.bias': (240,),
                    'critic.weight': (1,h), 'critic.bias': (1,)}
        if set(expected) != set(self.weights) or any(v.dtype != np.float64 or not np.isfinite(v).all()
            or v.shape != expected[k] for k,v in self.weights.items()):
            raise ValueError('finite shape-matched FP64 roll-in weights required')
        self.stochastic = stochastic; self.rng = np.random.default_rng(seed)
        self.calls = self.hits = 0; self.cache = OrderedDict()

    def logits(self, obs):
        encoded = encode(obs, allow_stop=False)
        # Exactly the policy's float32 encoded inputs promoted to float64.
        x = np.asarray(encoded['features'], dtype=np.float32).astype(np.float64)
        w = self.weights
        h = np.tanh(w['body.0.weight'] @ x + w['body.0.bias'])
        h = np.tanh(w['body.2.weight'] @ h + w['body.2.bias'])
        logits = w['actor.weight'] @ h + w['actor.bias']
        if not np.isfinite(logits).all():
            raise ValueError('nonfinite roll-in logits')
        return np.where(encoded['action_mask'], logits, -np.inf)

    def decide(self, obs):
        if not self.stochastic:
            return super().decide(obs)
        self.calls += 1
        mapping = action_map(obs, 'slots240_v1', allow_stop=False)
        logits = self.logits(obs); probabilities = np.exp(logits-np.max(logits))
        probabilities /= probabilities.sum()
        return dict(mapping[int(self.rng.choice(240, p=probabilities))])


class CorrectionSearch(Search):
    def __init__(self, seed, seconds, checkpoint, checkpoint_sha256, stochastic=False, **kwargs):
        super().__init__(seed, 'mixed_compiled', seconds, **kwargs)
        self.student = Student64(checkpoint, checkpoint_sha256, stochastic=stochastic, seed=seed+130)
        self.width = 7  # Preserve original six candidates plus the actual student action.
        self.student_action = None

    def rank_candidates(self, observation):
        choices = mixed_candidates(observation, 6)
        if action_key(self.student_action) not in {action_key(a) for _,a in choices}:
            choices.append((0., self.student_action))
        return choices

    def _decide(self, observation, context, begin):
        self.student_action = self.student.decide(observation)
        expert = super()._decide(observation, context, begin)
        self.last_search.update(expert_action=expert, student_action=self.student_action,
                                rollin='stochastic_model_only' if self.student.stochastic else 'greedy_model_only',
                                labels_only=True)
        candidates = self.last_search['candidates']
        student = next(c for c in candidates if c['action'] == self.student_action)
        selected = next(c for c in candidates if c['action'] == expert)
        self.last_search['teacher_minus_student_estimated_return'] = (
            selected['mean']-student['mean'] if student['samples'] >= 2 else None)
        return self.student_action


def make_policy(method, multiplier, search_seed, **params):
    if method != 'terminal' or multiplier != 1 or params.pop('mode') != 'timed' or params.pop('continuation') != 'dagger':
        raise ValueError('only train-time correction search allowed')
    return CorrectionSearch(search_seed, **params)


def run_episode(job, out, **kwargs):
    return base_episode(job, out, policy_factory=make_policy, log_pre_observation=True, **kwargs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); result = run_episode(read_json(a.job), a.out)
    print(json.dumps({k:result[k] for k in ('id','status','score','decision_seconds')}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)
