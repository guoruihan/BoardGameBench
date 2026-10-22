"""Preserve curriculum actor at transfer; learn a public-opponent residual."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from boardbench.contracts import Solver
from boardbench.environments.harmonies_duo import HarmoniesDuo, player_view
from boardbench.v138.baseline import Planner, build_model
from boardbench.v140.encoding import ACTIONS, BASE, FEATURES, action_map, encode_batch
from boardbench.v140.model import DuoPolicy
from boardbench.v142.policy import load_policy as load_market

LAYOUT = 'duo_full_public_curriculum_projection_opponent_residual_v143'
FORMAT = 'harmonies_duo_selfplay_policy_v143'


def own_projection(x):
    """Same input convention as V142, not the V142 transition/termination rules.

    The inherited turn13 indicator is just a public feature. Official final-round
    and both boards remain available unchanged to the new opponent branch.
    """
    meta=x[:,-8:];zero=torch.zeros_like(meta[:,0])
    turn13=((meta[:,2]*20).round()+1==13).to(x.dtype)
    own_meta=torch.stack((torch.ones_like(zero),turn13,meta[:,2],zero,meta[:,4],zero,meta[:,6],zero),-1)
    return torch.cat((x[:,:2*BASE],torch.zeros_like(x[:,2*BASE:3*BASE]),x[:,3*BASE:-8],own_meta),-1)


class SelfplayPolicy(nn.Module):
    def __init__(self, core):
        super().__init__()
        if core.objective!='score': raise ValueError('own-score curriculum core required')
        self.core=core
        self.opponent=nn.Sequential(nn.Linear(FEATURES,128),nn.Tanh(),nn.Linear(128,ACTIONS+1)).double()
        nn.init.zeros_(self.opponent[-1].weight);nn.init.zeros_(self.opponent[-1].bias)

    @property
    def solo(self): return self.core.solo  # Reuse existing PPO update diagnostics.

    @property
    def context(self): return self.opponent

    def forward(self,x,mask):
        logits,value=self.core(own_projection(x),mask)
        residual=self.opponent(x)
        return (logits+residual[:,:ACTIONS]).masked_fill(~mask,-1e9),value+residual[:,ACTIONS]


def initialize(parent,device='cpu'):
    return SelfplayPolicy(load_market(parent)).to(device)


def export_policy(model,path,config):
    value=dict(format=FORMAT,rules_version=HarmoniesDuo.rules_version,layout=LAYOUT,objective='margin',
        hidden=model.solo.actor.in_features,method=model.solo.method,strength=model.solo.strength,
        source_id=config.get('source_id'),parent_sha256=config['parent_sha256'],
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    torch.save(value,path);return value


def load_policy(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if (saved['format']!=FORMAT or saved['rules_version']!=HarmoniesDuo.rules_version
            or saved['layout']!=LAYOUT or saved['objective']!='margin'):
        raise ValueError('self-play artifact mismatch')
    model=SelfplayPolicy(DuoPolicy(build_model(saved['hidden'],saved['method'],saved['strength']),'score'))
    if any(v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('finite FP64 weights required')
    model.load_state_dict(saved['model'],strict=True);return model.to(device).eval()


class SelfplaySolver(Solver):
    def __init__(self,model=None,device='cpu'):
        self.model,self.device=model,device;self.planners=[Planner(),Planner()]

    def start_episode(self,observation,context=None): self.planners=[Planner(),Planner()]
    def load_weights(self,path): self.model=load_policy(path,self.device)

    def decide(self,observation,context=None):
        if observation['rules_version']!=HarmoniesDuo.rules_version or self.model is None:
            raise ValueError('official-duo observation and self-play model required')
        mapping=action_map(observation);seat=observation['active_player']
        x=encode_batch([observation],[self.planners[seat].vector(player_view(observation))])
        mask=np.array([[i in mapping for i in range(ACTIONS)]])
        with torch.inference_mode():
            logits,_=self.model(torch.as_tensor(x,dtype=torch.float64,device=self.device),torch.as_tensor(mask,device=self.device))
        return dict(mapping[int(logits.argmax(-1))])

    def save(self,directory): raise NotImplementedError('export_policy requires source/parent provenance')
    def load(self,directory): self.load_weights(Path(directory)/'weights.pt')
