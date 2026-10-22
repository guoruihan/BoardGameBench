"""Standalone fixed-panel evaluation. Confirmation never supplies training choices."""
import argparse
from pathlib import Path
from boardbench.artifacts.store import digest,read_json,write_json
from .training import Trainer,compact
from .policy import load_policy
from .evaluation import paired_delta


def run(spec,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=244000000 if spec['stage']=='development' else 344000000 if spec['stage']=='confirmation' else None
    if base is None or not spec['seeds'] or any(not base<=s<base+1000000 for s in spec['seeds']):
        raise ValueError('evaluation stage namespace mismatch')
    if digest(spec['model'])!=spec['model_sha256']:raise ValueError('evaluation policy digest mismatch')
    config=read_json(spec['config']);t=Trainer(config,device)
    try:
        model=load_policy(spec['model'],device);t.model.load_state_dict(model.state_dict())
        result=t.evaluate_panel(spec['seeds'])
        write_json(out/'result.json',dict(**result,stage=spec['stage'],model=spec['model'],model_sha256=spec['model_sha256']))
        if spec['stage']=='confirmation':
            t.model.load_state_dict(t.references['transfer'].state_dict());baseline=t.evaluate_panel(spec['seeds'])
            write_json(out/'baseline.json',baseline)
            write_json(out/'summary.json',dict(result=compact(result),baseline=compact(baseline),
                paired_vs_transfer=paired_delta(result,baseline),selection_uses_confirmation=False,
                selected_model=spec['model'],selected_sha256=spec['model_sha256']))
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(read_json(a.spec),a.out,a.device)
