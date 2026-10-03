"""Package source and real V1 evidence, excluding environments and unrelated runs."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from boardbench.artifacts.store import digest, source_id, write_json


def collect_tree(root, directories):
    files, empty_directories = [], []
    for directory in directories:
        tree = root/directory
        if not tree.is_dir():
            raise ValueError(f'missing required directory: {directory}')
        for path in [tree, *tree.rglob('*')]:
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_file():
                files.append(path)
            elif path.is_dir() and not any(path.iterdir()):
                empty_directories.append(str(path.relative_to(root)))
    return files, sorted(set(empty_directories))


def write_entries(archive, root, manifest, empty_directories):
    # Checkpoint/workspace may be empty but is a required restore directory.
    for relative in empty_directories:
        archive.writestr(relative.rstrip('/')+'/', b'')
    for relative in manifest:
        archive.write(root/relative, relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.out).resolve()
    if destination.exists():
        raise ValueError('archive exists; use a new filename')
    selected, empty_directories = collect_tree(root, [
        'src/boardbench', 'configs', 'tests', 'scripts', 'examples', 'docs', 'outputs/v1',
        'logs/v1', 'build/ui', 'runs/v1_browser_trained', 'runs/v1_portable'])
    for relative in ['.gitignore', 'README.md', 'IMPLEMENTATION_STATUS.md', 'pyproject.toml',
                     'CHANGES.md', 'CHANGELOG.md', 'runs/v1_acceptance_tests_final.xml',
                     'runs/v1_artifact_audit.json', 'runs/v1_wheel_validation.json',
                     'deliverables/v1_wheels_isolated/boardbench-0.2.0-py3-none-any.whl']:
        path = root/relative
        if not path.is_file():
            raise ValueError(f'missing required evidence: {relative}')
        selected.append(path)
    manifest = {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in sorted(set(selected))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        write_entries(archive, root, manifest, empty_directories)
        archive.writestr('V1_BUNDLE_MANIFEST.json', json.dumps({
            'source_id': source_id(), 'files': manifest, 'empty_directories': empty_directories}, indent=2))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for relative, expected in manifest.items():
            data = archive.read(relative)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
        assert all(archive.getinfo(relative+'/').is_dir() for relative in empty_directories)
    result = {'path': str(destination), 'bytes': destination.stat().st_size, 'files': len(manifest)+1,
              'empty_directories': len(empty_directories), 'sha256': digest(destination),
              'source_id': source_id(), 'all_members_verified': True}
    write_json(destination.with_suffix('.verification.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
