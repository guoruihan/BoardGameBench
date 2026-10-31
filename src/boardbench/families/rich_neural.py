"""Context-conditioned actor/critic for the versioned public tree interface.

Separate from legacy checkpoints/PPO. This module supplies executable forward,
gradient and batching contracts; the V172 phase does not train a new baseline.
"""
import numpy as np
import torch
from torch import nn

from .rich_observation import DEPTH, numeric_inputs


def tensors(rows,device='cpu'):
    return tuple(torch.as_tensor(x,device=device) for x in numeric_inputs(rows))


class TreePool(nn.Module):
    def __init__(self,vocabulary_size,hidden=128,embedding=8,hierarchical=False):
        super().__init__()
        self.symbol=nn.Embedding(vocabulary_size,embedding,padding_idx=0)
        self.kind=nn.Embedding(9,embedding)
        self.node=nn.Sequential(nn.Linear((DEPTH+2)*embedding+DEPTH+1,hidden),
                                nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.hierarchical=hierarchical
        if hierarchical:self.field=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Tanh())

    @staticmethod
    def pool(h,owners,count):
        sums=h.new_zeros((count,h.shape[-1])).index_add_(0,owners,h)
        sizes=torch.bincount(owners,minlength=count).clamp_min(1).to(h.dtype)
        maxima=h.new_full((count,h.shape[-1]),-torch.inf)
        maxima.scatter_reduce_(0,owners[:,None].expand_as(h),h,reduce='amax',include_self=True)
        maxima=torch.where(torch.isfinite(maxima),maxima,torch.zeros_like(maxima))
        return torch.cat((sums/sizes[:,None],maxima),-1)

    def forward(self,rows,owners,count):
        keys,kinds,symbols,numbers=rows
        h=self.node(torch.cat((self.symbol(keys).flatten(1),self.kind(kinds),
                               self.symbol(symbols),numbers),-1))
        if self.hierarchical:
            # A long static catalog/history must not wash out the board/hand.
            groups,inverse=torch.unique(torch.stack((owners,keys[:,0]),1),dim=0,return_inverse=True)
            h=self.field(self.pool(h,inverse,len(groups)))
            owners=groups[:,0]
        return self.pool(h,owners,count)


class ContextActorCritic(nn.Module):
    def __init__(self,vocabulary_size,hidden=128):
        super().__init__()
        self.hidden=hidden
        self.state=TreePool(vocabulary_size,hidden,hierarchical=True)
        self.action=TreePool(vocabulary_size,hidden)
        self.context_norm=nn.LayerNorm(2*hidden)
        self.choice_norm=nn.LayerNorm(2*hidden)
        self.actor=nn.Sequential(nn.Linear(6*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))

    def forward(self,state_rows,state_owners,action_rows,action_owners,action_states,state_count):
        context=self.context_norm(self.state(state_rows,state_owners,state_count))
        choices=self.choice_norm(self.action(action_rows,action_owners,len(action_states)))
        own=context[action_states]
        logits=self.actor(torch.cat((own,choices,own*choices),-1)).squeeze(-1)
        return logits,self.critic(context).squeeze(-1)


def batch_inputs(records,device='cpu'):
    """Ragged batching; no duplicated state per candidate or padded tree rows."""
    states=[];actions=[];state_owners=[];action_owners=[];action_states=[];offset=0
    for i,record in enumerate(records):
        s,a,owners=record['state'],record['actions'],record['owners']
        if len(s)==0 or len(owners)==0:raise ValueError('Nonempty state and legal actions required')
        count=int(owners[-1])+1
        if set(owners.tolist())!=set(range(count)):raise ValueError('Candidate alignment')
        states.append(s);actions.append(a);state_owners.extend([i]*len(s))
        action_owners.extend((owners+offset).tolist());action_states.extend([i]*count);offset+=count
    return (tensors(np.concatenate(states),device),torch.tensor(state_owners,device=device),
            tensors(np.concatenate(actions),device),torch.tensor(action_owners,device=device),
            torch.tensor(action_states,device=device),len(records))


FORMAT='boardbench_public_tree_actor_critic_v1'


def save_model(path,model,encoder,metadata=None):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(format=FORMAT,encoding=encoder.metadata,hidden=model.hidden,
                 metadata=metadata or {},model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    temporary=path.with_suffix('.part');torch.save(payload,temporary);temporary.replace(path)


def load_model(path,game,device='cpu'):
    from .rich_observation import PublicTreeEncoder
    encoder=PublicTreeEncoder(game);payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!=FORMAT or payload.get('encoding')!=encoder.metadata:
        raise ValueError('Rich model game/rules/vocabulary mismatch; legacy weights are not compatible')
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in payload['model'].values()):
        raise ValueError('Expected finite FP32 rich model weights')
    model=ContextActorCritic(encoder.metadata['vocabulary_size'],payload['hidden'])
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),encoder,payload


class RichPolicy:
    """Executable player interface; a newly initialized model has no strength claim."""
    def __init__(self,model,encoder,device='cpu'):
        self.model=model.to(device).eval();self.encoder=encoder;self.device=device

    def decide(self,observation,actions):
        if not actions:raise ValueError('Cannot choose from an empty legal-action set')
        rows,owners=self.encoder.encode_actions(actions)
        record=dict(state=self.encoder.encode(observation),actions=rows,owners=owners)
        with torch.inference_mode():logits,_=self.model(*batch_inputs([record],self.device))
        return actions[int(logits.argmax())]
