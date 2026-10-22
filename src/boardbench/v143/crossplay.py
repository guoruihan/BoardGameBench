"""Post-selection current-versus-league evaluation; never selects checkpoints."""
import argparse
from pathlib import Path
import time
from boardbench.artifacts.store import digest,read_json,write_json,source_id
from .policy import load_policy
from .training import Trainer


def run(config,model_a,model_b,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    trainer=Trainer(config,device)
    try:
        trainer.model=load_policy(model_a,device)
        trainer.frozen=load_policy(model_b,device).requires_grad_(False)
        result=trainer.evaluate(config['confirmation_seeds'])
        write_json(out/'result.json',dict(**result,source_id=source_id(),selection_uses_crossplay=False,
            model_a=str(model_a),model_b=str(model_b),model_a_sha256=digest(model_a),model_b_sha256=digest(model_b),
            wall_seconds=time.monotonic()-started))
    finally:trainer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','model-a','model-b','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(read_json(a.config),a.model_a,a.model_b,a.out)
