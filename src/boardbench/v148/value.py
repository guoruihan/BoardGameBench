"""Explicit calibrated remaining-score leaf model; no raw PPO critic fallback."""
import math

from boardbench.artifacts.store import read_json, digest
from boardbench.v145.policy import load_policy
from boardbench.v146.policy import ModelPlayer
from .search import policy_evaluation


class LeafValue:
    def __init__(self, manifest, sha256, native, expected_opponent):
        if digest(manifest)!=sha256: raise ValueError('immutable value manifest')
        m=read_json(manifest)
        if (m['format']!='v148_remaining_score_value' or not m['eligible_for_search']
                or m['target']!='remaining_own_score_div150'
                or expected_opponent not in m['opponent_hashes']
                or digest(m['model'])!=m['model_sha256']):
            raise ValueError('calibrated own-score value with matching opponent required')
        self.player=ModelPlayer(load_policy(m['model']),native)

    def __call__(self,obs,plans):
        seat=obs['active_player']
        _,_,remaining=policy_evaluation(self.player,obs,plans[seat])
        score=obs['score_breakdown'][seat]['total']+150*remaining
        if not math.isfinite(score): raise ValueError('nonfinite terminal score estimate')
        return score


def calibration(model, metrics, learner):
    # This is a predictive prerequisite, NOT evidence that leaf values improve
    # planning. Equal-budget paired gameplay is the separate research gate.
    eligible=(metrics['environments']>=8 and metrics['value_mse']<metrics['baseline_value_mse']*.75
              and metrics['value_mse']<.02)
    return dict(format='v148_remaining_score_value',target='remaining_own_score_div150',
                model=str(model),model_sha256=digest(model),eligible_for_search=eligible,
                metrics=metrics,opponent_hashes=learner.opponent_hashes,
                note='Held-out train-environment prediction gate only; candidate ranking/search gain remains empirical.')
