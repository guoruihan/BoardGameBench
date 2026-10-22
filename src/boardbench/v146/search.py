"""Cumulative-budget terminal rollout, exact duo model, public resampled futures."""
from copy import deepcopy
import math
import random
import time

from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from .policy import Habitat, mixed_roots


class TerminalSearch:
    def __init__(self, focal, opponent, seat, continuation, seconds=900., seed=0,
                 width=6, max_rounds=None, clock=time.monotonic):
        if (seat not in (0, 1) or continuation not in ('habitat', 'model') or
                not math.isfinite(seconds) or seconds <= 0 or not 1 <= width <= 32 or
                (max_rounds is not None and max_rounds < 1)):
            raise ValueError('invalid bounded search')
        self.players = [focal if i == seat else opponent for i in range(2)]
        self.seat, self.continuation, self.seconds = seat, continuation, seconds
        self.width, self.max_rounds, self.clock = width, max_rounds, clock
        self.rng = random.Random(seed); self.habitat = Habitat()
        self.spent = 0.; self.horizon = 75.; self.last = {}

    def terminal_round(self, obs, roots, seed, planners, deadline):
        branches = [FastHarmoniesDuo.from_observation(obs, seed) for _ in roots]
        memories = [deepcopy(planners) for _ in roots]
        states = []; own_counts = [1] * len(roots); steps = 0
        for branch, action in zip(branches, roots):
            if self.clock() >= deadline:
                return None, steps, []
            states.append(branch.step(action).observation); steps += 1
        for _ in range(768):
            active = [i for i, state in enumerate(states) if not state['terminated']]
            if not active:
                return [s['score_breakdown'][self.seat]['total'] for s in states], steps, own_counts
            if self.clock() >= deadline:
                return None, steps, []
            actions = {}
            for seat in (0, 1):
                slots = [i for i in active if states[i]['active_player'] == seat]
                if not slots:
                    continue
                if seat == self.seat and self.continuation == 'habitat':
                    for i in slots:
                        if self.clock() >= deadline:
                            return None, steps, []
                        actions[i] = self.habitat.decide(states[i])
                else:
                    ranks, _, _, _ = self.players[seat].rank_batch(
                        [states[i] for i in slots], [memories[i][seat] for i in slots])
                    actions.update({i: rank[0] for i, rank in zip(slots, ranks)})
            for i in active:
                if self.clock() >= deadline:
                    return None, steps, []
                own_counts[i] += states[i]['active_player'] == self.seat
                states[i] = branches[i].step(actions[i]).observation; steps += 1
        raise RuntimeError('duo rollout exceeded safety limit; no truncated leaf value')

    def decide(self, obs, planners):
        if obs['active_player'] != self.seat:
            raise ValueError('search may only choose for focal seat')
        begin = self.clock()
        rankings, _, _, _ = self.players[self.seat].rank_batch([obs], [planners[self.seat]])
        roots = mixed_roots(obs, rankings[0], self.habitat, self.width)
        # Reserve a small amount for future mandatory policy calls. All calls,
        # including fallback/candidates, remain in the measured decision budget.
        remaining = max(0., self.seconds-self.spent)
        allowance = max(0., remaining-.02*self.horizon) * min(1., 2/max(2., self.horizon))
        deadline = begin + allowance
        sums = [0.] * len(roots); squares = [0.] * len(roots)
        rounds = steps = 0; own_counts = []
        while (len(roots) > 1 and self.clock() < deadline and
               (self.max_rounds is None or rounds < self.max_rounds)):
            values, count, own = self.terminal_round(obs, roots, self.rng.getrandbits(63), planners, deadline)
            steps += count
            if values is None:
                break  # Incomplete candidate rounds never enter Q.
            for i, value in enumerate(values):
                sums[i] += value; squares[i] += value*value
            rounds += 1; own_counts.extend(own)
        selected = max(range(len(roots)), key=lambda i: sums[i]) if rounds else 0
        elapsed = self.clock()-begin; self.spent += elapsed
        self.last = dict(continuation=self.continuation,completed_rounds=rounds,simulation_steps=steps,
            selected_rank=selected,forced_action=len(roots)==1,fallback=not rounds and len(roots)>1,
            decision_seconds=elapsed,cumulative_decision_seconds=self.spent,public_future_only=True,
            candidates=[dict(action=a,samples=rounds,mean=sums[i]/rounds if rounds else None,
                variance=max(0.,(squares[i]-sums[i]**2/rounds)/(rounds-1)) if rounds>1 else None)
                for i,a in enumerate(roots)])
        if own_counts:
            self.horizon = max(1., .2*self.horizon+.8*sum(own_counts)/len(own_counts)-1)
        else:
            self.horizon = max(1., self.horizon-1)
        return dict(roots[selected])
