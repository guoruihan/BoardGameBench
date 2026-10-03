"""Package V1.1 source, actual models and raw evidence, preserving empty dirs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from package_v1 import collect_tree, write_entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    destination=Path(args.out).resolve()
    if destination.exists():
        raise ValueError('use a fresh archive filename')
    for relative in ('runs/v11_browser_animals_a/report.json',
                     'runs/v11_browser_trained_final/browser_check.json',
                     'runs/v11_portable/report.json', 'runs/v11_artifact_audit.json',
                     'runs/v11_policy_labels.json', 'runs/v11_wheel_validation.json'):
        if read_json(root / relative)['status'] != 'passed':
            raise ValueError(f'acceptance not passed: {relative}')
    selected,empty=collect_tree(root,[
        'src/boardbench','configs','tests','scripts','examples','docs','outputs/v11','logs/v11',
        'runs/v11_browser_animals_a','runs/v11_browser_trained','runs/v11_browser_trained_final',
        'runs/v11_portable','build/ui'])
    for relative in ['.gitignore','README.md','IMPLEMENTATION_STATUS.md','pyproject.toml',
                     'CHANGES.md','CHANGELOG.md','runs/v11_acceptance_final.xml',
                     'runs/v11_artifact_audit.json','runs/v11_wheel_validation.json','runs/v11_policy_labels.json',
                     'deliverables/v11_wheels/boardbench-0.3.0-py3-none-any.whl']:
        path=root/relative
        if not path.is_file():
            raise ValueError(f'missing required evidence: {relative}')
        selected.append(path)
    manifest={str(path.relative_to(root)):{'sha256':digest(path),'bytes':path.stat().st_size} for path in sorted(set(selected))}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'x',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        write_entries(archive,root,manifest,empty)
        archive.writestr('V11_BUNDLE_MANIFEST.json',json.dumps({'source_id':source_id(),'git_commit':commit,
                          'files':manifest,'empty_directories':empty},indent=2))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for relative,expected in manifest.items():
            data=archive.read(relative)
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
        assert all(archive.getinfo(relative+'/').is_dir() for relative in empty)
    result={'path':str(destination),'bytes':destination.stat().st_size,'files':len(manifest)+1,
            'empty_directories':len(empty),'sha256':digest(destination),'source_id':source_id(),
            'git_commit':commit,'all_members_verified':True}
    write_json(destination.with_suffix('.verification.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
