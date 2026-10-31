"""Prepare matched reward pilots from completed B runs and their training data."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import torch

from .neural import CandidateActorCritic, load_model, save_model
from .play import write_json
from .pool_training import file_hash
from .registry import game_class
from .reward_signals import SCORELESS, calibrate, validate_signal


def training_results(directory, count=32):
    results, provenance = [], []
    for path in sorted((Path(directory)/'trajectories').glob('batch_*.json*.gz')):
        with gzip.open(path, 'rt') as handle:
            if path.name.endswith('.jsonl.gz'):
                next(handle)  # batch header; only full TRAINING episodes follow
                for line in handle:
                    row = json.loads(line)['result']
                    if row['status'] == 'complete':
                        results.append(row)
                    if len(results) >= count:
                        break
            else:
                for episode in json.load(handle)['games']:
                    if episode['result']['status'] == 'complete':
                        results.append(episode['result'])
                    if len(results) >= count:
                        break
        provenance.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
        if len(results) >= count:
            break
    if len(results) < count:
        raise ValueError(f'Insufficient completed training games for calibration: {directory}')
    return results, provenance


def prepare(source, output, source_commit, games=None, calibration_games=32):
    source, output = Path(source).resolve(), Path(output).resolve()
    report = {}
    for old in sorted((source/'runs').glob('*/B')):
        cfg = json.loads((old/'config.json').read_text())
        game = cfg['game']
        if games and game not in games:
            continue
        state = json.loads((old/'state.json').read_text())
        if state['phase'] not in ('complete', 'budget_exhausted'):
            raise ValueError('Parent experiment must finish before preparing an intervention')
        target = output/'inputs'/game
        target.mkdir(parents=True, exist_ok=False)
        results, archives = training_results(old, calibration_games)
        schema = calibrate(results)
        schema['provenance'] = archives
        write_json(target/'calibration.json', schema)
        parent = json.loads((old/'latest.json').read_text())
        if parent['kind'] != 'neural' or file_hash(parent['path']) != parent['sha256']:
            raise ValueError('Expected a checksum-verified common neural parent')
        cls = game_class(game)
        original, saved = load_model(parent['path'], cls)
        torch.manual_seed(1711)
        model = CandidateActorCritic(original.hidden, schema['names'])
        missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
        if set(missing) != {'auxiliary.weight', 'auxiliary.bias'} or unexpected:
            raise ValueError('Parent architecture does not support explicit auxiliary expansion')
        initial = target/'initial.pt'
        provenance = dict(kind='frozen_B_final_plus_auxiliary_heads', parent=parent,
                          parent_source=cfg['source_commit'], calibration=archives)
        save_model(initial, model, cls, saved['prior_weights'], provenance)
        panel = deepcopy(cfg['panel'])
        panel['initial_common'] = dict(kind='neural', path=str(initial), sha256=file_hash(initial))
        # Parent training throughput determines a conservative COMMON decision cap.
        decisions = max(100_000, min(500_000, int(state['learner_decisions']*.4)//10_000*10_000))
        modes = ['win', 'aux'] + ([] if game in SCORELESS else ['mixed'])
        for mode in modes:
            run = output/'runs'/game/mode
            run.mkdir(parents=True, exist_ok=False)
            current = {k:deepcopy(v) for k,v in cfg.items() if k not in (
                'budget_origin_deadline', 'resume_provenance')}
            signal = dict(mode=mode, schema=schema, auxiliary_coef=0. if mode=='win' else .25,
                          mix_weight=.25 if mode=='mixed' else 0.)
            validate_signal(signal, game, model.auxiliary_names)
            current.update(output=str(run), source_commit=source_commit, arm='B', condition=mode,
                initial_path=str(initial), initial_sha256=file_hash(initial), initial_provenance=provenance,
                panel=panel, frozen_training_opponents=deepcopy(panel),
                seed=1711, workers=4, games=24, hours=1., deadline_unix=0., max_updates=4096,
                max_decisions=decisions, eval_decisions=max(25000, decisions//2),
                quick_seeds=16, confirm_seeds=128, reward_signal=signal,
                reward='terminal_win' if mode!='mixed' else '0.75*win+0.25*tanh(final_margin/train_margin_std)',
                network='candidate_mlp_64_64_optional_terminal_heads',
                comparison='matched parents/opponents/seeds/decision caps; report achieved counts and timeouts')
            write_json(run/'config.json', current)
        report[game] = dict(conditions=modes, initial_sha256=file_hash(initial),
                           max_decisions=decisions, auxiliary_names=schema['names'],
                           skipped_mixed_reason='no_native_victory_point_score' if game in SCORELESS else None)
    if not report:
        raise ValueError('No qualified completed parents')
    write_json(output/'prepared.json', dict(source_commit=source_commit, source=str(source), games=report,
               jobs=sum(len(x['conditions']) for x in report.values()),
               max_condition_hours=sum(len(x['conditions']) for x in report.values())))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--games', nargs='*')
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.source_commit, args.games), indent=2))


if __name__ == '__main__':
    main()
