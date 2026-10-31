"""Read local pilot artifacts without loading models or running evaluations."""
import argparse
import json
from pathlib import Path


def report(directory):
    rows=[]
    for config in sorted(Path(directory).glob('runs/*/*/config.json')):
        cfg=json.loads(config.read_text());out=config.parent
        state=json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {}
        records=[]
        if (out/'metrics.jsonl').exists():
            for line in (out/'metrics.jsonl').read_text().splitlines():
                try:records.append(json.loads(line))
                except json.JSONDecodeError:pass  # A live writer may have one incomplete last line.
        validations=[r for r in records if r.get('event')=='validation']
        completed=[r for r in validations if r['summary']['all_complete']]
        best=max(completed,key=lambda r:r['summary']['quality'],default=None)
        baseline=state.get('baseline') or {}
        row=dict(game=cfg['game'],arm=cfg['arm'],phase=state.get('phase','not_started'),
            updates=state.get('next_update',0),learner_decisions=state.get('learner_decisions',0),
            initial_quality=baseline.get('quality'),latest_quality=validations[-1]['summary']['quality'] if validations else None,
            highest_quick_quality=best['summary']['quality'] if best else None,
            highest_quick_candidate=best['candidate'] if best else None,
            confirmed_best=state.get('best'),costs=state.get('costs'),error=state.get('error'),
            exit_code=(out/'exit_code').read_text().strip() if (out/'exit_code').exists() else None,
            note='development pilot; highest quick score is not a confirmed promotion')
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();print(json.dumps(report(a.directory),indent=2))


if __name__=='__main__':main()
