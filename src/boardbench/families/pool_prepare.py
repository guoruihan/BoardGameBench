"""Prepare common-method A/B runs from a checksum-verified frozen bot bundle."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .bundle import load_bundle
from .neural import CandidateActorCritic, load_model, save_model
from .play import write_json
from .pool_training import file_hash
from .registry import game_class


def prepare(bundle, output, source_commit, games=None):
    root, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = load_bundle(root, verify_code=False)  # Rules/checksums still mandatory; new trainer code differs.
    prepared = {}
    for game, entry in manifest['games'].items():
        if entry['status']!='selected' or games and game not in games:
            continue
        cls = game_class(game)
        directory = output/'inputs'/game
        if directory.exists():
            raise ValueError('Prepare requires a new run directory; never overwrite frozen input')
        directory.mkdir(parents=True)
        specs = {}
        for name, bot in entry['bots'].items():
            spec_path = root/bot['path']
            spec = json.loads(spec_path.read_text())
            if 'path' in spec:
                origin = (spec_path.parent/spec['path']).resolve()
                copied = directory/(name+'.pt')
                shutil.copyfile(origin, copied)
                if file_hash(copied)!=spec['sha256']:
                    raise ValueError('Copied model checksum mismatch')
                spec['path'] = str(copied)
            specs[name] = spec
        compatible = [name for name in entry['recommended'] if specs[name]['kind']=='neural']
        compatible += [name for name in sorted(specs) if specs[name]['kind']=='neural' and name not in compatible]
        initial_path = directory/'initial.pt'
        if compatible:
            name = compatible[0]
            model, saved = load_model(specs[name]['path'], cls)
            if model.hidden != 64:
                raise ValueError('Unified baseline requires the common 64-wide network')
            shutil.copyfile(specs[name]['path'], initial_path)
            provenance = dict(kind='common_rl_continuation', bot=name, sha256=file_hash(initial_path),
                              bundle_sha256=manifest['manifest_sha256'])
        else:
            # The generic fallback applies to every game without compatible
            # common weights. No Harmonies-only network, planner or loss.
            torch.manual_seed(1701)
            model = CandidateActorCritic(64)
            save_model(initial_path, model, cls, cls.default_weights,
                       dict(initialization='shared_heuristic_prior_zero_actor', source_commit=source_commit))
            name = None
            provenance = dict(kind='common_heuristic_prior_initialization', bot=None,
                              reason='no_compatible_common_checkpoint', bundle_sha256=manifest['manifest_sha256'])
        initial = dict(kind='neural', path=str(initial_path), sha256=file_hash(initial_path))
        panel = {'initial_common':initial}
        alternatives = [n for n in entry['recommended'] if n!=name]
        alternatives += [n for n in sorted(specs) if n not in (name, 'random', 'heuristic') and n not in alternatives]
        if alternatives:
            panel[alternatives[0]] = specs[alternatives[0]]
        panel['heuristic'] = specs['heuristic']
        write_json(directory/'panel.json', panel)
        for arm in ('A', 'B'):
            run = output/'runs'/game/arm
            run.mkdir(parents=True)
            cfg = dict(format='boardbench_policy_pool_pilot_v1', game=game, arm=arm, output=str(run),
                source_commit=source_commit, initial_path=str(initial_path), initial_sha256=file_hash(initial_path),
                initial_provenance=provenance, panel=panel, device='cuda', seed=1701,
                workers=6, games=24, hours=1., deadline_unix=0., max_updates=4096,
                max_decisions=2_000_000, eval_decisions=20_000, max_match_seconds=120.,
                quick_seeds=32, confirm_seeds=128, gamma=1., gae_lambda=.95,
                learning_rate=1e-4, entropy_coef=.01, prior_kl_coef=.01, epochs=4, batch_size=128,
                observation='public_candidate_features_24', network='candidate_mlp_64_64',
                reward='terminal_win_plus1_draw0_loss_minus1', incomplete='exclude_whole_learner_chain',
                comparison='pilot_walltime_cap; compare learner-decision counts, not update count')
            write_json(run/'config.json', cfg)
        prepared[game] = dict(initial_provenance=provenance, initial_sha256=file_hash(initial_path), panel=list(panel))
    write_json(output/'prepared.json', dict(source_commit=source_commit, games=prepared))
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--games', nargs='*')
    a = p.parse_args()
    print(json.dumps(prepare(a.bundle, a.output, a.source_commit, a.games), indent=2))


if __name__=='__main__':
    main()
