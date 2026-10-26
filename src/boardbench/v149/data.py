"""Reuse audited old value labels; select a single teacher without re-encoding."""
import argparse
from pathlib import Path

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json


def select_teacher(manifest, out, method='search_habitat'):
    out=Path(out)
    if (out/'manifest.json').exists():
        existing=read_json(out/'manifest.json')
        if existing['parent_manifest_sha256']!=digest(manifest) or existing['selected_teacher']!=method:
            raise ValueError('immutable dataset configuration changed')
        return existing
    meta=read_json(manifest)
    if meta['format']!='v148_search_training' or meta['split']!='train' or digest(meta['path'])!=meta['sha256']:
        raise ValueError('audited immutable V148 training data required')
    ids=[];games=[];cursor=0
    for game in meta['games']:
        if game['method']==method:
            ids.extend(range(cursor,cursor+game['rows']));games.append(game)
        cursor+=game['rows']
    if cursor!=meta['rows'] or not ids:raise ValueError('dataset row ordering mismatch or no teacher')
    with np.load(meta['path'],allow_pickle=False) as f:
        arrays={k:f[k][ids] for k in f.files}
    expected=[(g['seed'],g['seat']) for g in games for _ in range(g['rows'])]
    if list(zip(arrays['seed'].tolist(),arrays['seat'].tolist()))!=expected:
        raise ValueError('dataset receipt/row alignment failed')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'data.npz';tmp=out/'data.partial.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    result={**meta,'path':str(path.resolve()),'sha256':digest(path),'rows':len(ids),'games':games,
            'environments':len(set(arrays['seed'].tolist())),
            'mcts_rows':int((arrays['label']==2).sum()),'legacy_rows':int((arrays['label']==1).sum()),
            'parent_manifest_sha256':digest(manifest),'selected_teacher':method}
    write_json(out/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();select_teacher(a.manifest,a.out)
