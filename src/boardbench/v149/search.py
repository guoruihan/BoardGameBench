"""Matched roots and public futures; one-step, turn-cutoff, or terminal evaluation.

Turn-cutoff is a root search with a fixed habitat continuation inside the turn,
not a jointly optimized macro-action or a full MCTS. Only complete paired sample
rounds are used; all candidate/encoding/inference work is charged to the call.
"""
from copy import deepcopy
import math
import signal
import threading
import time

import numpy as np

from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from boardbench.v146.policy import mixed_roots
from boardbench.v146.search import TerminalSearch


class DecisionDeadline(Exception):
    pass


class Search(TerminalSearch):
    def __init__(self, focal, opponent, seat, continuation, seconds=.1, seed=0,
                 *, value=None, width=6, max_rounds=None, clock=time.monotonic):
        if continuation not in ('rollout', 'value_one', 'value_short', 'value_turn'):
            raise ValueError('unknown V149 evaluation method')
        if not math.isfinite(seconds) or not 0 < seconds <= 1:
            raise ValueError('per-call budget must be in (0, 1] seconds')
        if continuation.startswith('value') and value is None:
            raise ValueError('value artifact required')
        super().__init__(focal, opponent, seat, 'habitat', seconds, seed,
                         width=width, max_rounds=max_rounds, clock=clock)
        self.mode, self.value = continuation, value

    def short_round(self, obs, roots, seed, planners, deadline):
        states, memories = [], []
        steps = 0
        turn = obs['players'][self.seat]['completed_turns']
        for action in roots:
            if self.clock() >= deadline:
                return None, steps
            game = FastHarmoniesDuo.from_observation(obs, seed)
            plans = deepcopy(planners)
            state = game.step(action).observation
            own_steps=1
            steps += 1
            for _ in range(768):
                if self.clock() >= deadline:
                    return None, steps
                if state['terminated']:
                    break
                seat = state['active_player']
                if seat == self.seat:
                    if (self.mode == 'value_one' or (self.mode=='value_short' and own_steps>=2)
                            or state['players'][seat]['completed_turns'] > turn):
                        break
                    action = self.habitat.decide(state)
                else:
                    action = self.players[seat].decide(state, plans[seat])
                state = game.step(action).observation
                own_steps+=seat==self.seat
                steps += 1
            else:
                raise RuntimeError('short rollout exceeded action bound')
            states.append(state)
            memories.append(plans)
        if self.clock() >= deadline:
            return None, steps
        values = self.value.batch(states, memories, self.seat)
        return (values if self.clock() < deadline else None), steps

    def decide(self, obs, planners):
        """Main-process timer also interrupts expensive Python candidate work.

        A delayed native call / OS descheduling can still overrun: measure it and
        fail the timing gate rather than pretending it met the wall-clock bound.
        """
        begin=self.clock()
        if obs['active_player']!=self.seat or obs['terminated']:
            raise ValueError('live focal decision required')
        self._safe_action=dict(obs['legal_actions'][0])
        self._timed_result={}
        timer=(self.clock is time.monotonic and threading.current_thread() is threading.main_thread())
        if timer:
            previous=signal.getsignal(signal.SIGALRM)
            if signal.getitimer(signal.ITIMER_REAL)[0]:raise ValueError('nested decision timer')
            def expired(*_):raise DecisionDeadline()
            signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.seconds-.003))
        interrupted=False
        try:
            action=self._decide(obs,planners)
        except DecisionDeadline:
            interrupted=True
            action=self._safe_action
            self.last={**dict(algorithm='v149_paired_root_search',cutoff=self.mode,
                budget_scope='per_actual_atomic_decision',budget_seconds=self.seconds,
                completed_rounds=0,simulation_steps=0,candidates=[],fallback=True,
                forced_action=False),**self._timed_result}
        finally:
            if timer:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,previous)
        elapsed=self.clock()-begin
        self.spent+=elapsed if interrupted else elapsed-self.last['decision_seconds']
        self.last.update(decision_seconds=elapsed,cumulative_decision_seconds=self.spent,
                         budget_excess_seconds=max(0.,elapsed-self.seconds),deadline_interrupted=interrupted)
        return action

    def _decide(self, obs, planners):
        if obs['active_player'] != self.seat or obs['terminated']:
            raise ValueError('live focal decision required')
        begin = self.clock()
        # Buffer for one uninterruptible inference/heuristic call and serialization.
        deadline = begin + max(0., self.seconds-max(.008, .15*self.seconds))
        roots = [dict(obs['legal_actions'][0])]
        rounds = steps = attempts = 0
        sums = squares = np.zeros(1)
        if len(obs['legal_actions']) > 1 and self.clock() < deadline:
            ranking = self.players[self.seat].rank_batch([obs], [planners[self.seat]])[0][0]
            roots = [ranking[0]]
            self._safe_action=dict(roots[0])
            if hasattr(self,'on_update'):self.on_update(self._safe_action,{})
            if self.clock() < deadline:
                roots = mixed_roots(obs, ranking, self.habitat, self.width)
            sums, squares = np.zeros(len(roots)), np.zeros(len(roots))
            while len(roots)>1 and self.clock()<deadline and (self.max_rounds is None or attempts<self.max_rounds):
                attempts += 1
                seed = self.rng.getrandbits(63)
                if self.mode == 'rollout':
                    values, count, _ = self.terminal_round(obs, roots, seed, planners, deadline)
                else:
                    values, count = self.short_round(obs, roots, seed, planners, deadline)
                steps += count
                if values is None or self.clock() >= deadline:
                    break
                values = np.asarray(values)
                sums += values
                squares += values*values
                rounds += 1
                self._safe_action=dict(roots[int(sums.argmax())])
                self._timed_result=dict(completed_rounds=rounds,simulation_steps=steps,
                    candidates=[dict(action=a,samples=rounds,mean=float(sums[i]/rounds),
                        variance=float(max(0.,(squares[i]-sums[i]**2/rounds)/(rounds-1))) if rounds>1 else None)
                        for i,a in enumerate(roots)],fallback=False)
                if hasattr(self,'on_update') and (rounds==1 or rounds%4==0):
                    self.on_update(self._safe_action,self._timed_result)
                # Before end_turn there is no chance event in a one-step successor.
                if self.mode == 'value_one' and all(a['type']!='end_turn' for a in roots):
                    break
        chosen = int(sums.argmax()) if rounds else 0
        elapsed = self.clock()-begin
        self.spent += elapsed
        self.last = dict(algorithm='v149_paired_root_search',cutoff=self.mode,
            budget_scope='per_actual_atomic_decision',budget_seconds=self.seconds,
            decision_seconds=elapsed,cumulative_decision_seconds=self.spent,
            budget_excess_seconds=max(0.,elapsed-self.seconds),simulation_steps=steps,
            completed_rounds=rounds,simulation_attempts=attempts,forced_action=len(obs['legal_actions'])==1,
            fallback=not rounds and len(obs['legal_actions'])>1,
            candidates=[dict(action=a,samples=rounds,mean=float(sums[i]/rounds) if rounds else None,
                variance=float(max(0.,(squares[i]-sums[i]**2/rounds)/(rounds-1))) if rounds>1 else None)
                for i,a in enumerate(roots)])
        return dict(roots[chosen])
