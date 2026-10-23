"""Pure model paired evaluation; independent confirmation never selects a model."""
import argparse
from pathlib import Path
from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v144.training import compact
from boardbench.v144.evaluation import paired_delta
from boardbench.v145.policy import load_policy
from .training import Trainer


def run(spec,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=247000000 if spec['stage']=='development' else 347000000 if spec['stage']=='confirmation' else None
    if base is None or not spec['seeds'] or any(not base<=s<base+1000000 for s in spec['seeds']):raise ValueError('panel namespace')
    if digest(spec['model'])!=spec['model_sha256']:raise ValueError('model hash')
    config={**read_json(spec['config']),'evaluation_deadline_unix':spec['deadline_unix'],
            'expert_manifest':None,'expert_weight':0.,'sil_weight':0.}
    t=Trainer(config,device)
    try:
        t.model.load_state_dict(load_policy(spec['model'],device).state_dict());result=t.evaluate_panel(spec['seeds'])
        write_json(out/'result.json',dict(**result,stage=spec['stage'],model=spec['model'],model_sha256=spec['model_sha256']))
        if spec['stage']=='confirmation':
            if digest(spec['baseline'])!=spec['baseline_sha256']:raise ValueError('baseline hash')
            t.model.load_state_dict(load_policy(spec['baseline'],device).state_dict());parent=t.evaluate_panel(spec['seeds'])
            write_json(out/'parent.json',parent)
            write_json(out/'summary.json',dict(result=compact(result),parent=compact(parent),
                paired_vs_parent=paired_delta(result,parent),selection_uses_confirmation=False,
                selected_model=spec['model'],selected_sha256=spec['model_sha256']))
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(read_json(a.spec),a.out,a.device)
