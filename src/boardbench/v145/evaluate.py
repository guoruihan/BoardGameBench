"""Fixed official-duo panels; fresh confirmation does not choose a model."""
import argparse
from pathlib import Path
from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v144.training import compact
from boardbench.v144.evaluation import paired_delta
from .training import Trainer
from .policy import load_policy


def run(spec,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=245000000 if spec['stage']=='development' else 345000000 if spec['stage']=='confirmation' else None
    if base is None or not spec['seeds'] or any(not base<=s<base+1000000 for s in spec['seeds']):raise ValueError('panel namespace mismatch')
    if digest(spec['model'])!=spec['model_sha256']:raise ValueError('evaluation digest mismatch')
    c={**read_json(spec['config']),'evaluation_deadline_unix':spec['deadline_unix']};t=Trainer(c,device)
    try:
        t.model.load_state_dict(load_policy(spec['model'],device).state_dict());result=t.evaluate_panel(spec['seeds'])
        write_json(out/'result.json',dict(**result,stage=spec['stage'],model=spec['model'],model_sha256=spec['model_sha256']))
        if spec['stage']=='confirmation':
            t.model.load_state_dict(t.training_anchor.state_dict());parent=t.evaluate_panel(spec['seeds'])
            t.model.load_state_dict(t.references['transfer'].state_dict());transfer=t.evaluate_panel(spec['seeds'])
            write_json(out/'parent.json',parent);write_json(out/'transfer.json',transfer)
            write_json(out/'summary.json',dict(result=compact(result),parent=compact(parent),transfer=compact(transfer),
                paired_vs_parent=paired_delta(result,parent),paired_vs_transfer=paired_delta(result,transfer),
                selection_uses_confirmation=False,selected_model=spec['model'],selected_sha256=spec['model_sha256']))
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(read_json(a.spec),a.out,a.device)
