"""Audited weight reuse for the two documented forced-pass rule corrections.

This is not a general schema override. Only Splendor/Gizmos rules 1 -> 2 are
supported. Old scores remain old-rule results; the copied policy needs new
development evaluation before it can be selected for the corrected runtime.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .play import FeaturePolicy
from .registry import game_class


def migrate_policy(game, spec, destination):
    cls = game_class(game)
    if game not in ('splendor', 'gizmos') or cls.rules_version != '2':
        raise ValueError('No audited migration for this game/runtime')
    if any(spec.get(k) != v for k, v in
           (('game', game), ('config', cls.config_id), ('rules_version', '1'))):
        raise ValueError('Expected an explicitly identified rules-1 policy')
    kind = spec.get('kind')
    if kind not in ('linear', 'neural') or spec.get('decision_guard'):
        raise ValueError('Only unguarded trained linear/neural policies can migrate')
    destination = Path(destination)
    if destination.suffix != '.json':
        raise ValueError('Destination must be a new JSON policy specification')
    model_path = destination.with_suffix('.pt')
    audit_path = destination.with_suffix('.migration.json')
    if any(p.exists() for p in (destination, model_path, audit_path)):
        raise FileExistsError('Migration never overwrites existing artifacts')
    legacy_cls = type('LegacyRulesIdentity', (cls,), {'rules_version': '1'})
    result = deepcopy(spec)
    result['rules_version'] = '2'
    audit = dict(format='boardbench_forced_pass_migration_v1', game=game,
                 source_rules_version='1', target_rules_version='2',
                 source_spec_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True,
                     allow_nan=False).encode()).hexdigest(),
                 inference_parameters_changed=False,
                 evaluation_required=True,
                 reason='Documented forced pass only when no normal action exists; '
                        'Splendor additionally discloses the reserved cards as ruled. '
                        'Existing 24-dimensional action features retain their meanings.')
    if kind == 'neural':
        import torch
        from .neural import load_model
        source = Path(spec['path'])
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if spec.get('sha256', source_hash) != source_hash:
            raise ValueError('Source model checksum mismatch')
        _, saved = load_model(source, legacy_cls)
        audit['source_model_sha256'] = source_hash
        copied = deepcopy(saved)
        copied['rules_version'] = '2'
        copied['metadata']['rules_migration'] = deepcopy(audit)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix('.part')
        torch.save(copied, temporary)
        # Validate exactly what was serialized, including finite tensor checks.
        _, checked = load_model(temporary, cls)
        if checked['prior_weights'] != saved['prior_weights'] or any(
                not torch.equal(v, checked['model'][k]) for k, v in saved['model'].items()):
            temporary.unlink()
            raise RuntimeError('Migration altered inference parameters')
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            temporary.unlink()
            raise RuntimeError('Source model changed during migration')
        temporary.replace(model_path)
        result.update(path=str(model_path.resolve()),
                      sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
        audit['target_model_sha256'] = result['sha256']
    else:
        FeaturePolicy(legacy_cls, spec)
        FeaturePolicy(cls, result)
        destination.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((audit_path, audit), (destination, result)):
        with path.open('x') as f:
            json.dump(value, f, allow_nan=False, indent=2)
    return result, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True, choices=['splendor', 'gizmos'])
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.policy.read_text())
    if spec.get('kind') == 'neural' and not Path(spec['path']).is_absolute():
        spec['path'] = str((a.policy.parent / spec['path']).resolve())
    _, audit = migrate_policy(a.game, spec, a.output)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
