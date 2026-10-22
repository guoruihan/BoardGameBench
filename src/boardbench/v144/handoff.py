"""Preserve stopped V143 best policies and a fixed development-selected opponent."""
import argparse
from pathlib import Path
import torch
from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v143.policy import initialize,export_policy


def run(previous,out):
    previous=Path(previous);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    status=read_json(previous/'status.json')
    if status['phase'] not in ('interrupted','completed'):raise ValueError('previous controller must stop before handoff')
    records=[]
    for trial in sorted((previous/'trials').iterdir()):
        config=read_json(trial/'config.json');latest=read_json(trial/'latest.json')
        if digest(trial/'latest.pt')!=latest['sha256']:raise ValueError('latest checkpoint inconsistent')
        selected=torch.load(trial/'best.pt',map_location='cpu',weights_only=False)
        if digest(trial/'best.pt')!=read_json(trial/'best.json')['sha256']:raise ValueError('best checkpoint inconsistent')
        model=initialize(config['parent']);model.load_state_dict(selected['model'])
        path=out/(trial.name+'.pt');export_policy(model,path,config)
        batch=selected['batches'];ev=read_json(trial/(f'dev_{batch:06d}.json' if batch else 'initial.json'))
        records.append(dict(name=trial.name,path=str(path.resolve()),sha256=digest(path),format='v143',
            selected_batch=batch,latest_batch=latest['batches'],selected_development=selected['best_metric'],
            score=ev['score_mean'],animal=ev['animal_mean'],source_id=config['source_id']))
    chosen=max(records,key=lambda r:r['selected_development'])
    write_json(out/'handoff.json',dict(previous=str(previous.resolve()),policies=records,
        evaluation_opponent={**chosen,'name':'v143_margin_snapshot'},
        choice='highest prior DEVELOPMENT margin; not a confirmation-selected opponent',all_old_artifacts_preserved=True))
    print(read_json(out/'handoff.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.previous,a.out)
