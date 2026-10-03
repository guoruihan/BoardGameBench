"""Sequential independent jobs on one explicitly reserved/visible GPU."""
import json
from pathlib import Path
import subprocess
import sys


for job in sys.argv[1:]:
    game,algorithm,seed=job.split(':')
    if game not in ('micro_tiles','take_it_easy','harmonies') or algorithm not in ('ppo','imitation') or seed not in ('811','812','813'):
        raise ValueError('job outside predeclared experiment matrix')
    config=Path('outputs/v11/experiment_plan')/f'{game}_{algorithm}_{seed}.json'
    output=Path('outputs/v11')/game/f'{algorithm}_{seed}'
    if output.exists():
        raise ValueError(f'refusing to overwrite {output}')
    module='boardbench.training' if algorithm=='ppo' else 'boardbench.imitation'
    command=[sys.executable,'-u','-m',module,'--config',str(config),'--out',str(output)]
    print(json.dumps({'job':job,'status':'starting','command':command}),flush=True)
    subprocess.run(command,check=True)
    print(json.dumps({'job':job,'status':'finished'}),flush=True)
