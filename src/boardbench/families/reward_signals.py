"""Terminal training labels, never additional policy observations or game rules.

Calibration uses completed TRAINING games only. Race progress is an auxiliary
outcome, not a fabricated victory-point score. Evaluation always uses real wins.
"""
import math

import numpy as np


MODES = ('win', 'aux', 'mixed')
SCORELESS = frozenset(('machi_koro', 'el_dorado'))


def numeric_leaves(value, prefix=''):
    """Stable named components; variable-length lists are summed, not indexed."""
    if isinstance(value, dict):
        result = {}
        for key, child in sorted(value.items()):
            result.update(numeric_leaves(child, f'{prefix}.{key}' if prefix else key))
        return result
    if isinstance(value, list):
        if all(type(x) in (int, float) and math.isfinite(x) for x in value):
            return {prefix + '.sum': float(sum(value))}
        return {}
    if type(value) in (int, float) and math.isfinite(value):
        return {prefix: float(value)}
    return {}


def raw_labels(result, seat):
    if seat not in (0, 1):
        raise ValueError('Expected learner seat 0 or 1')
    if result['status'] != 'complete':
        return {}
    labels = {}
    scores = result.get('outcome', {}).get('scores', [])
    if (len(scores) == 2 and result['game'] not in SCORELESS and
            all(type(s) in (int, float) and math.isfinite(s) for s in scores)):
        labels.update(own_score=float(scores[seat]), opponent_score=float(scores[1-seat]),
                      score_margin=float(scores[seat]-scores[1-seat]))
    breakdown = result.get('score_breakdown', [])
    if len(breakdown) == 2:
        for who, player in (('own', seat), ('opponent', 1-seat)):
            for key, value in numeric_leaves(breakdown[player]).items():
                if key != 'total':
                    labels[f'{who}.{key}'] = value
    return labels


def calibrate(results):
    complete = [r for r in results if r['status'] == 'complete']
    if not complete or len({r['game'] for r in complete}) != 1:
        raise ValueError('Calibration requires completed training games of one game')
    rows = [raw_labels(r, seat) for r in complete for seat in (0, 1)]
    names = sorted(set().union(*(set(r) for r in rows)))
    if not names:
        raise ValueError('No terminal targets available')
    means, scales = [], []
    for name in names:
        values = [r[name] for r in rows if name in r]
        means.append(float(np.mean(values)))
        scales.append(max(1., float(np.std(values))))
    margin = names.index('score_margin') if 'score_margin' in names else None
    return dict(format='boardbench_terminal_labels_v1', game=complete[0]['game'],
                names=names, means=means, scales=scales, clip=10.,
                margin_scale=scales[margin] if margin is not None else None,
                calibration_games=len(complete), calibration_seeds=[r['seed'] for r in complete])


def validate_signal(config, game, names=None):
    if config['mode'] not in MODES or config['schema']['game'] != game:
        raise ValueError('Reward mode/game mismatch')
    schema = config['schema']
    n = len(schema['names'])
    if (not n or len(set(schema['names'])) != n or
            len(schema['means']) != n or len(schema['scales']) != n or
            not all(math.isfinite(x) for x in schema['means']) or
            not all(math.isfinite(x) and x > 0 for x in schema['scales']) or
            not math.isfinite(schema['clip']) or schema['clip'] <= 0):
        raise ValueError('Invalid immutable target calibration')
    if names is not None and list(names) != schema['names']:
        raise ValueError('Model/target column order differs')
    coefficient = config.get('auxiliary_coef', 0.)
    weight = config.get('mix_weight', 0.)
    if (not math.isfinite(coefficient) or coefficient < 0 or
            not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('Invalid reward/loss coefficient')
    if config['mode'] == 'win' and (coefficient != 0 or weight != 0):
        raise ValueError('Win control cannot enable extra losses or rewards')
    if config['mode'] == 'aux' and weight != 0:
        raise ValueError('Auxiliary supervision must preserve win reward')
    if config['mode'] == 'mixed':
        scale = schema['margin_scale']
        if (game in SCORELESS or 'score_margin' not in schema['names'] or
                scale is None or not math.isfinite(scale) or scale <= 0 or weight <= 0):
            raise ValueError('Mixed score objective requires genuine final scores')


def terminal_signals(result, seat, config):
    if result['status'] != 'complete':
        raise ValueError('Incomplete episodes cannot supply terminal supervision')
    validate_signal(config, result['game'])
    raw, schema = raw_labels(result, seat), config['schema']
    targets, mask, clipped = [], [], 0
    for name, mean, scale in zip(schema['names'], schema['means'], schema['scales']):
        present = name in raw
        value = (raw[name]-mean)/scale if present else 0.
        clipped += int(present and abs(value) > schema['clip'])
        targets.append(float(np.clip(value, -schema['clip'], schema['clip'])))
        mask.append(present)
    win = float(result['rewards'][seat])
    reward = win
    if config['mode'] == 'mixed':
        weight = config['mix_weight']
        reward = (1-weight)*win + weight*math.tanh(raw['score_margin']/schema['margin_scale'])
    return dict(targets=targets, mask=mask, clipped=clipped, win_reward=win,
                training_reward=reward)
