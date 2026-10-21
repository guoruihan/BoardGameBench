"""Package committed source and portable weights; no training history required."""
import argparse
import io
from pathlib import Path
import subprocess
import tarfile

from boardbench.artifacts.store import digest, read_json, write_json


def package(artifact, out):
    artifact, out = Path(artifact), Path(out)
    if out.exists(): raise FileExistsError(out)
    manifest = read_json(artifact/'manifest.json')
    if digest(artifact/'weights.pt') != manifest['sha256']: raise ValueError('weights digest mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source = subprocess.check_output(['git', 'archive', '--format=tar', 'HEAD'])
    with tarfile.open(out, 'w:gz') as archive:
        with tarfile.open(fileobj=io.BytesIO(source)) as original:
            for entry in original:
                stream = original.extractfile(entry) if entry.isfile() else None
                entry.name = 'boardgame/'+entry.name
                archive.addfile(entry, stream)
        for name in ('weights.pt', 'manifest.json'):
            archive.add(artifact/name, arcname='boardgame/baseline/'+name)
    write_json(out.with_suffix(out.suffix+'.json'), dict(path=out.name, sha256=digest(out),
               source_commit=commit, weights_sha256=manifest['sha256'], kind='source_plus_inference_weights'))
    print(out, digest(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); package(args.artifact, args.out)
