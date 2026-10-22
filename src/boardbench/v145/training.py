"""Official-duo true-score PPO, stationary opponents, plan credit and own SIL."""
import argparse
from copy import deepcopy
import time
import numpy as np
import torch
from boardbench.artifacts.store import digest,read_json
from boardbench.v133.training import Replay
from boardbench.v140.training import infer_actions,summarize
from boardbench.v144.training import Trainer as PreviousTrainer,run as previous_run
from .policy import LAYOUT,load_policy,export_policy
from .credit import plan_potential,credit,visible_cards


class Trainer(PreviousTrainer):
    VERSION=145
    LAYOUT=LAYOUT
    OBJECTIVES=('score',)
    MODES=('current','fixed','mixed')
    DEVELOPMENT_BASE=245000000
    CONFIRMATION_BASE=345000000
    load_warm_policy=staticmethod(load_policy)
    export_policy=staticmethod(export_policy)

    def __init__(self,config,device='cpu'):
        super().__init__(config,device)
        if not 0<=config['progress_scale']<=2 or not 0<=config['sil_weight']<=.2:
            raise ValueError('bounded learning interventions required')
        if digest(config['training_anchor'])!=config['training_anchor_sha256']:
            raise ValueError('immutable stationary opponent digest mismatch')
        self.training_anchor=load_policy(config['training_anchor'],device).eval().requires_grad_(False)
        self.replay=(Replay(config['replay_directory'],config['replay_batches'],config['seed']+145000)
            if config['sil_weight'] else None)

    def checkpoint_extra_state(self):
        return dict(**super().checkpoint_extra_state(),replay=self.replay.state() if self.replay else None)

    def restore_extra_state(self,state):
        super().restore_extra_state(state)
        if (state.get('replay') is None)!=(self.replay is None):raise ValueError('replay configuration changed')
        if self.replay:
            seeds={s for r in state['replay']['receipts'] for s in r['environment_seeds']}
            if any(not 145000000<=s<146000000 for s in seeds):raise ValueError('replay split leakage')
            self.replay.restore(state['replay'],seeds)

    def collect(self,pool,seeds,learner_seats,opponents,greedy=False,keep_rows=True):
        if not len(seeds)==len(learner_seats)==len(opponents)==pool.count:raise ValueError('one role per match')
        if keep_rows and any(not 145000000<=s<146000000 for s in seeds):raise ValueError('training-only rows/replay')
        tick=time.monotonic();arrays=pool.reset(seeds);env_seconds=time.monotonic()-tick;model_seconds=0.
        calls=total_actions=0;trajectories=[[[],[]] for _ in seeds];traces=[[] for _ in seeds]
        tokens=np.zeros((len(seeds),2),int);animals=np.zeros_like(tokens);cards=np.zeros_like(tokens)
        seen=[[set(),set()] for _ in seeds];prefixes=[[{},{}] for _ in seeds]
        models={'learner':self.model,'current':self.model,'frozen':self.frozen}
        while not arrays['done'].all():
            slots=np.flatnonzero(~arrays['done']);seats=arrays['stats'][slots,4].astype(int)
            xs=arrays['features'][slots].copy();masks=arrays['masks'][slots].copy()
            roles=['learner' if seat==learner_seats[slot] else opponents[slot] for slot,seat in zip(slots,seats)]
            tick=time.monotonic();inferred,n=infer_actions(models,roles,xs,masks,self.device,self.rng,greedy)
            actions=inferred[:,0].astype(np.int64);calls+=n;total_actions+=len(slots);model_seconds+=time.monotonic()-tick
            phi=plan_potential(xs) if keep_rows and self.config['progress_scale'] else np.zeros(len(slots))
            for j,slot in enumerate(slots):
                seat=int(seats[j]);a=int(actions[j]);role=roles[j]
                tokens[slot,seat]+=3<=a<141;animals[slot,seat]+=144<=a<236;cards[slot,seat]+=a in (141,142,143,242,243)
                seen[slot][seat].update(visible_cards(xs[j]))
                if keep_rows:
                    trainable=role in ('learner','current')
                    traces[slot].append(dict(seat=seat,action=a,logp=float(inferred[j,1]),value=float(inferred[j,2]),
                        role=role,trainable=trainable,plan_potential=float(phi[j])))
                    if trainable:
                        trajectories[slot][seat].append((xs[j],masks[j],a,float(inferred[j,1]),float(inferred[j,2]),
                            float(arrays['stats'][slot,seat]),len(traces[slot])-1,float(phi[j])))
            tick=time.monotonic();pool.step(slots,actions);env_seconds+=time.monotonic()-tick
            for j,slot in enumerate(slots):
                if actions[j]==236:
                    seat=int(seats[j]);turn=int(round(float(xs[j,-6])*20))+1
                    if turn in (8,10,11):prefixes[slot][seat][turn]=(float(arrays['stats'][slot,seat]),float(arrays['stats'][slot,2+seat]))
        games=[];rows=[];seat_counts=[0,0];learners=0;archived=[]
        for slot,(seed,focus,opponent) in enumerate(zip(seeds,learner_seats,opponents)):
            s=arrays['stats'][slot]
            game=dict(seed=int(seed),seat=int(focus),opponent=opponent,score=float(s[focus]),opponent_score=float(s[1-focus]),
                animal=float(s[2+focus]),opponent_animal=float(s[3-focus]),margin=float(s[focus]-s[1-focus]),
                win_credit=float(s[5] if focus==0 else 1-s[5]),natural_terminal=True,turns=int(s[6])-1,
                tokens=int(tokens[slot,focus]),animals_placed=int(animals[slot,focus]),cards_taken=int(cards[slot,focus]),
                visible_unique_cards=len(seen[slot][focus]))
            for turn in (8,10,11):
                score,animal=prefixes[slot][focus].get(turn,(game['score'],game['animal']))
                game.update({f'prefix{turn}_score':score,f'prefix{turn}_animal':animal,f'prefix{turn}_reached':turn in prefixes[slot][focus]})
            games.append(game)
            if keep_rows:
                learned=[]
                for seat,trajectory in enumerate(trajectories[slot]):
                    if not trajectory:continue
                    learners+=1;learned.append(seat);seat_counts[seat]+=len(trajectory)
                    scores=[r[5] for r in trajectory]+[float(s[seat])]
                    raw=np.diff(scores)/150
                    shaped,adv,targets,mc=credit(raw,[r[4] for r in trajectory],[r[7] for r in trajectory],
                        self.config['progress_scale'],self.config['gae_lambda'])
                    if abs(raw.sum()*150-float(s[seat]))>1e-7:raise RuntimeError('own-score return mismatch')
                    for i,r in enumerate(trajectory):
                        traces[slot][r[6]].update(reward_to_next_own_decision=float(raw[i]),training_reward=float(shaped[i]),
                            mc_training_return=float(mc[i]))
                        rows.append((*r[:4],float(targets[i]),float(adv[i]),float(mc[i])))
                archived.append(dict(**game,learner_seats=learned,opponent_snapshot_batch=None,
                    progress_scale=self.config['progress_scale'],decisions=traces[slot]))
        if keep_rows:
            self.last_traces=archived
            if self.replay:self.replay.add(rows,self.actions,seeds)
        return rows,dict(**summarize(games),per_game=games,environment_seconds=env_seconds,model_seconds=model_seconds,
            total_player_actions=total_actions,learner_actions_by_seat=seat_counts,learner_trajectories=learners,
            current_selfplay_matches=opponents.count('current'),fixed_training_matches=opponents.count('frozen'),
            inference_calls=calls,mean_inference_batch=total_actions/max(1,calls))

    def update(self,rows):
        result=super().update(rows);losses=[]
        if self.replay:
            for _ in range(self.config['sil_steps']):
                sample=self.replay.sample(self.config['sil_minibatch'])
                x=torch.as_tensor(sample['x'],dtype=torch.float64,device=self.device)
                mask=torch.as_tensor(sample['mask'],device=self.device);a=torch.as_tensor(sample['action'],device=self.device)
                target=torch.as_tensor(sample['mc'],dtype=torch.float64,device=self.device)
                logits,value=self.model(x,mask);positive=(target-value).clamp_min(0)
                if not (positive>0).any():continue
                weights=positive.detach().clamp_max(1.)
                loss=self.config['sil_weight']*((-torch.distributions.Categorical(logits=logits).log_prob(a)*weights).mean()
                    +.5*positive.square().mean())
                if not torch.isfinite(loss):raise RuntimeError('nonfinite SIL')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5,error_if_nonfinite=True);self.optimizer.step()
                losses.append(float(loss.detach()))
        return dict(**result,sil_loss=float(np.mean(losses)) if losses else 0.,sil_updates=len(losses),
            replay_rows=sum(len(b['action']) for b in self.replay.blocks) if self.replay else 0)

    def batch(self):
        self.get_pool(self.config['batch_matches']);seeds=self.seed_rng.integers(145000000,146000000,self.pool.count).tolist()
        seats=[(i//4+self.batches)%2 for i in range(self.pool.count)]
        opponents=['current' if self.config['mode']=='current' or (self.config['mode']=='mixed' and i%2==0)
            else 'frozen' for i in range(self.pool.count)]
        evaluation_anchor=self.frozen;self.frozen=self.training_anchor
        try:rows,result=self.collect(self.pool,seeds,seats,opponents)
        finally:self.frozen=evaluation_anchor
        result.update(self.update(rows));self.batches+=1;self.matches+=len(seeds);self.actions+=len(rows)
        return dict(**result,batches=self.batches,matches=self.matches,actions=self.actions,
            snapshot_batches=[],objective='score',progress_scale=self.config['progress_scale'],sil_weight=self.config['sil_weight'])

    def evaluate_panel(self,seeds):
        result=super().evaluate_panel(seeds)
        for key in ('cards_taken','visible_unique_cards',*[f'prefix{t}_{k}' for t in (8,10,11) for k in ('score','animal','reached')]):
            result[key+'_mean']=float(np.mean([g[key] for g in result['per_game']]))
        return result


def run(config,out,device='cuda',resume=False):
    return previous_run(config,out,device,resume,trainer_class=Trainer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
