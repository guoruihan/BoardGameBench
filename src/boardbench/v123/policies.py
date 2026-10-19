"""Fixed diagnostic continuations; feature bonuses are not game rewards.

space adds empty-cell/stack opportunity costs to the old heuristic. habitat adds
reachable, compatible one-completion goals, not exhaustive sequential planning.
"""
from collections import Counter, OrderedDict
from itertools import product
import hashlib
import json

from boardbench.environments.harmonies import CARDS, COLORS, PATTERNS, STACKS, STACK_INDEX
from boardbench.solvers.board import ranked_actions

POLICIES = ('original', 'space', 'habitat')


def available_tokens(obs):
    """Public optimistic upper bound, not guaranteed future resource access.

    Unchosen bundles become inaccessible after choosing. Before choosing, all
    three are included: optimism is intentional and documented, never clairvoyance.
    """
    available = Counter(obs['remaining_tokens']) + Counter(obs['pending_tokens'])
    if not obs['bundle_taken']:
        available.update(color for bundle in obs['token_market'] for color in bundle or [])
    return available


def reachable_pattern(board, anchor, pattern, available):
    """Cheapest exact target assignment respecting prefix/lock/aggregate supply.

    Occupied supporting cells are fine when matching; occupied anchors are not.
    """
    if board[anchor]['animal'] is not None:
        return None
    alternatives = []
    for cell, allowed in pattern:
        current = STACKS[board[cell]['stack_id']]
        options = []
        for target in allowed:
            desired = STACKS[target]
            if desired[:len(current)] != current:
                continue
            missing = desired[len(current):]
            if missing and board[cell]['animal'] is not None:
                continue
            options.append((cell, target, missing))
        if not options:
            return None
        alternatives.append(options)
    best = None
    for choice in product(*alternatives):
        need = Counter(color for _, _, missing in choice for color in missing)
        if any(n > available[color] for color, n in need.items()):
            continue
        key = (sum(need.values()), tuple(target for _, target, _ in choice))
        if best is None or key < best[0]:
            best = (key, {'anchor': anchor, 'targets': {cell: target for cell, target, _ in choice},
                          'need': dict(need), 'missing': sum(need.values())})
    return None if best is None else best[1]


def card_goals(obs, card_id, placed_count=0):
    available, goals = available_tokens(obs), []
    track = CARDS[card_id]['score_by_placed_count']
    marginal = track[placed_count + 1] - track[placed_count]
    for anchor, orientations in enumerate(PATTERNS[card_id]):
        for orientation, pattern in orientations.items():
            goal = reachable_pattern(obs['board'], anchor, pattern, available)
            if goal is not None:
                goal.update(card_id=card_id, orientation=orientation,
                            value=marginal / (1 + goal['missing']))
                goals.append(goal)
    return sorted(goals, key=lambda g: -g['value'])


def compatible_goals(obs, by_card):
    """Greedy one completion/card, share exact supporting targets, distinct anchors.

    Incompatible simultaneous goals are omitted, not declared globally impossible.
    """
    chosen, used_cards, targets, anchors = [], set(), {}, set()
    available = available_tokens(obs)
    for goal in sorted((g for goals in by_card.values() for g in goals), key=lambda g: -g['value']):
        if goal['card_id'] in used_cards or goal['anchor'] in anchors:
            continue
        if any(cell in targets and targets[cell] != target for cell, target in goal['targets'].items()):
            continue
        merged = {**targets, **goal['targets']}
        need = Counter(color for cell, target in merged.items()
                       for color in STACKS[target][len(STACKS[obs['board'][cell]['stack_id']]):])
        if any(count > available[color] for color, count in need.items()):
            continue
        chosen.append(goal)
        used_cards.add(goal['card_id'])
        anchors.add(goal['anchor'])
        targets = merged
    return chosen


def extension_count(cell):
    if cell['animal'] is not None:
        return 0
    current = STACKS[cell['stack_id']]
    return sum(current + (color,) in STACK_INDEX for color in COLORS)


def rank_actions(obs, policy, *, goal_function=None):
    goal_function = card_goals if goal_function is None else goal_function
    if policy not in POLICIES:
        raise ValueError('unknown diagnostic continuation')
    ranked = ranked_actions(obs)
    if policy == 'original':
        return ranked
    board = obs['board']
    empty = sum(c['stack_id'] == 0 for c in board)
    future_turns = sum(obs['remaining_tokens'].values()) // 9
    space_price = min(2.5, 3 * future_turns / max(1, empty - 2)) if future_turns else 0.
    by_card, goals = {}, []
    if policy == 'habitat':
        for card in obs['active_cards']:
            by_card[card['card_id']] = goal_function(obs, card['card_id'], card['placed_count'])
        goals = compatible_goals(obs, by_card)
    market_goals, scored = {}, []
    for base, action in ranked:
        kind, value = action['type'], base
        if kind == 'place_token':
            cell = action['cell_id']
            old = board[cell]['stack_id']
            new = STACK_INDEX[STACKS[old] + (action['color'],)]
            value += -.8 * space_price if old == 0 else .25 * space_price
            for goal in goals:
                target = goal['targets'].get(cell)
                if target is None:
                    continue
                desired, actual = STACKS[target], STACKS[new]
                if desired[:len(actual)] == actual:
                    value += min(3., goal['value']) / max(1, goal['missing'])
                else:
                    value -= min(4., goal['value'])
        elif kind == 'place_animal':
            anchor = action['anchor_cell_id']
            value -= .2 * space_price * extension_count(board[anchor])
            for goal in goals:
                if goal['card_id'] != action['instance_id'] and (goal['anchor'] == anchor or
                    (anchor in goal['targets'] and goal['targets'][anchor] != board[anchor]['stack_id'])):
                    value -= min(4., goal['value'])
        elif kind == 'take_card' and policy == 'habitat':
            card_id = obs['animal_market'][action['market_slot']]
            if card_id not in market_goals:
                market_goals[card_id] = goal_function(obs, card_id)
            candidates = market_goals[card_id]
            # A genuinely unreachable card must lose to end_turn (-1), otherwise
            # the limited active slots fill with permanently unusable cards.
            value = -3. if not candidates else .2 + candidates[0]['value']
            if candidates:
                combined = compatible_goals(obs, {**by_card, card_id: candidates})
                if not any(g['card_id'] == card_id for g in combined):
                    value -= 1.
        elif kind == 'choose_bundle':
            bundle = obs['token_market'][action['bundle_id']]
            for color in bundle:
                if any(c['stack_id'] and c['animal'] is None and
                       STACKS[c['stack_id']] + (color,) in STACK_INDEX for c in board):
                    value += .25 * space_price
            targets = {cell: target for g in goals for cell, target in g['targets'].items()}
            needed = Counter(color for cell, target in targets.items()
                             for color in STACKS[target][len(STACKS[board[cell]['stack_id']]):])
            needed.subtract(obs['pending_tokens'])
            value += .8 * sum(min(n, max(0, needed[c])) for c, n in Counter(bundle).items())
        scored.append((value, action))
    return sorted(scored, key=lambda pair: -pair[0])  # ties retain original ranking


class Continuation:
    def __init__(self, policy, capacity=8192):
        if policy not in POLICIES or capacity < 0:
            raise ValueError('invalid policy/cache')
        self.policy, self.capacity = policy, capacity
        self.cache = OrderedDict()
        self.calls = self.hits = 0

    def decide(self, obs):
        self.calls += 1
        key = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode()).digest()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        action = rank_actions(obs, self.policy)[0][1]
        if self.capacity:
            self.cache[key] = dict(action)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return dict(action)
