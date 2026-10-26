"""Own-return components and a separate habitat-continuation ranking head."""
import numpy as np
import torch
from torch import nn

from boardbench.artifacts.store import digest,read_json
from boardbench.environments.harmonies_duo import player_view
from boardbench.v140.encoding import FEATURES,encode_batch
from boardbench.v145.policy import load_policy
from boardbench.v149.value import remaining_value,Value as LegacyValue


class Heads(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,hidden),nn.Tanh(),nn.Linear(hidden,3)).double()
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base):
        residual=self.net(x)
        components=base[...,None]*.5+residual[...,:2]  # terrain, animal; /150 each
        rank=x[...,-4]+base+residual[...,2]  # final own score /150 under habitat continuation
        return components,rank


def load_head(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='v150_component_rank_head':raise ValueError('head format')
    head=Heads(saved['hidden']);head.load_state_dict(saved['state'])
    if any(not torch.isfinite(v).all() for v in head.state_dict().values()):raise ValueError('nonfinite head')
    return head.to(device).eval()


class Value:
    def __init__(self,manifest,sha,native,opponent_hash,rank_mix=0.):
        if digest(manifest)!=sha or not 0<=rank_mix<=1:raise ValueError('manifest/ranking mix')
        m=read_json(manifest)
        if (m['format']!='v150_component_value' or m['target']!='remaining_own_terrain_animal_div150'
                or opponent_hash not in m['opponent_hashes']):raise ValueError('component target/opponent')
        for p,h in [('base_model','base_sha256'),('head','head_sha256')]:
            if digest(m[p])!=m[h]:raise ValueError('immutable value weights')
        self.base=load_policy(m['base_model']).requires_grad_(False)
        self.head=load_head(m['head']);self.native=native;self.mix=rank_mix;self.metadata=m

    def batch(self,observations,memories,seat):
        result=[None]*len(observations);live=[]
        for i,obs in enumerate(observations):
            if obs['terminated']:result[i]=float(obs['scores'][seat])
            elif obs['active_player']!=seat:raise ValueError('focal decision required')
            else:live.append(i)
        if live:
            obs=[observations[i] for i in live]
            plans=[memories[i][seat].vector(player_view(observations[i])) for i in live]
            x=torch.as_tensor(encode_batch(obs,plans,self.native),dtype=torch.float64)
            with torch.inference_mode():
                base=remaining_value(self.base,x);parts,rank=self.head(x,base)
                own=torch.tensor([o['score_breakdown'][seat]['total']/150 for o in obs],dtype=torch.float64)
                values=150*((1-self.mix)*(own+parts.sum(-1))+self.mix*rank)
            for i,v in zip(live,values.tolist()):result[i]=v
        if not np.isfinite(result).all():raise ValueError('nonfinite planning score')
        return result


def make_value(path,sha,native,opponent,rank_mix=0.):
    if read_json(path)['format']=='v148_remaining_score_value':
        if rank_mix:raise ValueError('legacy value has no ranking head')
        return LegacyValue(path,sha,native,opponent)
    return Value(path,sha,native,opponent,rank_mix)
