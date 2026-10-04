"""Explicit research partitions; final test requires a frozen selection artifact."""
import argparse
from pathlib import Path

from boardbench.artifacts.store import read_json, write_json, digest, source_id
from boardbench.v12.evaluation import evaluate
from .registry import validate_manifest, identity


def policy(config, weights):
    return {'id': f'{config["experiment_id"]}_{config["training_seed"]}_{Path(weights).stem}',
            'solver': {'id': 'harmonies_research', 'params': {**config['network'], 'device': 'cpu'}},
            'weights': str(Path(weights).resolve()), 'weights_sha256': digest(weights),
            'training_seed': config['training_seed'], 'experiment_id': config['experiment_id']}


def seeds_for(manifest, split, spec, frozen=None):
    validate_manifest(manifest)
    if split not in ('development_health', 'development', 'confirmation', 'test', 'historical_audit'):
        raise ValueError('unknown research evaluation partition')
    if split == 'test':
        if not frozen or frozen.get('split_sha256') != manifest['sha256'] or frozen.get('source_id') != source_id():
            raise ValueError('final test requires source/split-matched frozen selection')
        if frozen.get('policy_hashes', {}).get(spec['id']) != identity(spec):
            raise ValueError('policy not in frozen selection')
        if frozen.get('selection_split') != 'confirmation' or not frozen.get('selection_rule'):
            raise ValueError('freeze must declare confirmation and fixed selection rule')
    return manifest['sets'][split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config');source.add_argument('--policy',help='explicit frozen baseline import')
    parser.add_argument('--weights')
    parser.add_argument('--splits', required=True); parser.add_argument('--split', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--frozen-selection')
    args = parser.parse_args()
    manifest = read_json(args.splits)
    if args.config:
        config = read_json(args.config)
        if not args.weights or config['split_sha256'] != manifest['sha256']:
            raise ValueError('weights required and training/evaluation splits must match')
        spec = policy(config, args.weights)
    else:
        imported=read_json(args.policy)
        if imported['split_sha256']!=manifest['sha256'] or imported['runtime_source_id']!=source_id():
            raise ValueError('explicit baseline import split/source mismatch')
        spec=imported['policy']
    frozen = read_json(args.frozen_selection) if args.frozen_selection else None
    seeds = seeds_for(manifest, args.split, spec, frozen)
    summary = evaluate('harmonies', spec, seeds, 3., args.out, args.workers)
    rows = [__import__('json').loads(x) for x in (Path(args.out)/'episodes.jsonl').read_text().splitlines()]
    from statistics import mean
    extended = {**summary, 'research_split': args.split, 'split_sha256': manifest['sha256'],
                'stop_rate': mean(r['reason'] == 'stopped' for r in rows),
                'timeout_rate': mean(r['reason'] == 'deadline' for r in rows),
                'failure_rate': mean(r['failed'] for r in rows),
                'actions_mean': mean(r['actions'] for r in rows)}
    write_json(Path(args.out) / 'research_summary.json', extended)
    if summary['failures']:
        raise SystemExit('evaluation failures retained; no capability claim')


if __name__ == '__main__':
    main()
