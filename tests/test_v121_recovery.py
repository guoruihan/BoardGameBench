from pathlib import Path
import shutil

import pytest

from boardbench.artifacts.store import digest, read_json, source_files, source_id, source_root, write_json
from boardbench.v121.recovery import numeric_parent_config, prepare_recovery
from boardbench.v121.registry import NUMERIC, make_splits, training_config
from boardbench.v121.training import Trainer


def test_progress_only_selection_includes_failed_phase_recovery_and_is_hash_checked(tmp_path):
    parent, new = tmp_path / 'parent', tmp_path / 'new'
    for relative in source_files():
        dest = parent / 'source' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root() / relative, dest)
    experiments = ['E00_T16_F64', 'E11_T16_F64']
    write_json(parent / 'allocation.json', {'source_id':source_id(), 'experiments':[NUMERIC[x] for x in experiments], 'training_seeds':[911]})
    write_json(parent / 'status.json', {'source_id':source_id(), 'phase':'failed', 'gpu_seconds':100,
        'cpu_core_seconds':850, 'elapsed_seconds':60, 'current_target_actions':10000})
    write_json(parent / 'watchdog_result.json', {'returncode':1})
    write_json(parent / 'learning_curve.json', [])
    splits = make_splits(train_count=32)
    for folder in (parent, new):
        write_json(folder / 'plan/splits.json', splits)
    selected = {}
    for name in experiments:
        cfg = training_config(name, 911, splits)
        cfg.update(batch_episodes=2, epochs=1, purpose='recovery_check')
        old_cfg = numeric_parent_config(cfg)
        write_json(new / 'plan/configs' / f'{name}_911.json', cfg)
        write_json(parent / 'plan/configs' / f'{NUMERIC[name]}_911.json', old_cfg)
        trainer = Trainer(old_cfg); trainer.ppo_batch()
        trainer.save(parent / 'training' / f'{NUMERIC[name]}_911_a1000_chunk00001/final.resume.pt')
        trainer.ppo_batch()
        recent = parent / 'training' / f'{NUMERIC[name]}_911_a10000_chunk00002/latest.resume.pt'
        trainer.save(recent)  # Failed phase has no summary/final, but a complete rolling checkpoint.
        selected[f'{name}_911'] = recent
        trainer.close()
    result = prepare_recovery(parent, new, experiments, [911])
    assert result['prior_gpu_seconds'] == 100
    for name, entry in result['entries'].items():
        assert Path(entry['resume']) == selected[name]
        assert entry['parent_sha256'] == digest(selected[name])
        assert entry['numeric_import']
    meta_path = next(iter(selected.values())).with_suffix('.json')
    meta = read_json(meta_path); meta['resume_sha256'] = 'corrupted'
    write_json(meta_path, meta)
    with pytest.raises(ValueError, match='checkpoint hash'):
        prepare_recovery(parent, new, experiments, [911])
