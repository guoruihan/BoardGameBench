import importlib.util
from pathlib import Path
from zipfile import ZipFile

from boardbench.artifacts.store import file_hashes
from boardbench.runner.engine import run, evaluate

spec = importlib.util.spec_from_file_location('package_v1', Path(__file__).resolve().parents[1]/'scripts/package_v1.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_zip_preserves_empty_checkpoint_workspace_and_can_restore(tmp_path):
    run_dir = tmp_path/'run'
    result = run({'task': {'id': 'micro_tiles', 'version': 'micro_tiles_v1'},
                  'solver': {'id': 'board', 'params': {'method': 'random'}},
                  'episodes': 1, 'checkpoint': {'every_episodes': 1}}, run_dir)
    checkpoint = run_dir/result['checkpoints'][0]
    assert (checkpoint/'workspace').is_dir() and not list((checkpoint/'workspace').iterdir())
    files, directories = packager.collect_tree(checkpoint, ['workspace', 'solver', 'source'])
    files.append(checkpoint/'manifest.json')
    manifest = {str(p.relative_to(checkpoint)): None for p in files}
    assert 'workspace' in directories
    with ZipFile(tmp_path/'bundle.zip', 'x') as z:
        packager.write_entries(z, checkpoint, manifest, directories)
    restored = tmp_path/'restored'
    with ZipFile(tmp_path/'bundle.zip') as z:
        z.extractall(restored)
    assert (restored/'workspace').is_dir()
    assert file_hashes(checkpoint) == file_hashes(restored)
    summary = evaluate(restored, {'episode_seeds': [123]}, tmp_path/'evaluated')
    assert summary['completed_episodes'] == 1


def test_packager_ignores_bytecode_not_empty_runtime_directories(tmp_path):
    (tmp_path/'payload/workspace').mkdir(parents=True)
    (tmp_path/'payload/__pycache__').mkdir()
    files, directories = packager.collect_tree(tmp_path, ['payload'])
    assert files == [] and directories == ['payload/workspace']
