"""Versioned objectives with unchanged public-state architecture and game rules."""
from pathlib import Path
import torch
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v138.baseline import build_model
from boardbench.v140.model import DuoPolicy
from boardbench.v143.policy import SelfplayPolicy,SelfplaySolver as PreviousSolver

LAYOUT='duo_full_public_curriculum_projection_opponent_residual_v144'
FORMAT='harmonies_duo_own_score_policy_v144'


def export_policy(model,path,config):
    value=dict(format=FORMAT,rules_version=HarmoniesDuo.rules_version,layout=LAYOUT,
        objective=config['objective'],hidden=model.solo.actor.in_features,
        method=model.solo.method,strength=model.solo.strength,source_id=config.get('source_id'),
        parent_sha256=config['parent_sha256'],warm_start_sha256=config.get('warm_start_sha256'),
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    path=Path(path);tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);tmp.replace(path)
    return value


def load_policy(path,device='cpu'):
    s=torch.load(path,map_location='cpu',weights_only=True)
    if (s['format']!=FORMAT or s['rules_version']!=HarmoniesDuo.rules_version
            or s['layout']!=LAYOUT or s['objective'] not in ('score','margin')):
        raise ValueError('V144 policy schema mismatch')
    if any(v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in s['model'].values()):
        raise ValueError('finite FP64 weights required')
    model=SelfplayPolicy(DuoPolicy(build_model(s['hidden'],s['method'],s['strength']),'score'))
    model.load_state_dict(s['model'],strict=True);return model.to(device).eval()


class SelfplaySolver(PreviousSolver):
    def load_weights(self,path):self.model=load_policy(path,self.device)
