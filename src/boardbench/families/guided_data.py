"""Trusted local expert records for public-state imitation/value initialization."""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import zlib

import numpy as np

from .guided_neural import base_policy, encode_record
from .play import write_json
from .reward_signals import calibrate, raw_labels
from .rich_observation import PublicTreeEncoder

GAMES = ('century_a_new_world', 'century_spice_road', 'clever', 'patchwork',
         'splendor', 'stone_age', 'harmonies', 'toy_battle')


def dump_packet(path, value):
    """Internal, locally generated cache; never load packets from untrusted users."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = zlib.compress(pickle.dumps(value, protocol=5), level=1)
    temporary = path.with_suffix('.part'); temporary.write_bytes(data); temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def load_packet(path, expected_hash=None):
    data = Path(path).read_bytes()
    if expected_hash and hashlib.sha256(data).hexdigest()!=expected_hash:
        raise ValueError('Local packet checksum mismatch')
    return pickle.loads(zlib.decompress(data))


def result_labels(result):
    result = dict(result)
    winners = result['outcome']['winners']
    result['rewards'] = [0., 0.] if len(winners)!=1 else [1. if p in winners else -1. for p in (0, 1)]
    return result


def targets(result, seat, schema):
    labels = raw_labels(result, seat)
    auxiliary = []
    for name, mean, scale in zip(schema['names'], schema['means'], schema['scales']):
        auxiliary.append(float(np.clip((labels.get(name, mean)-mean)/scale, -10., 10.)))
    mask = [name in labels for name in schema['names']]
    win = float(result['rewards'][seat])
    margin = np.tanh(labels.get('score_margin', 0.)/max(1., schema.get('margin_scale') or 1.))
    i = schema['names'].index('own_score') if 'own_score' in schema['names'] else None
    own = np.tanh((labels['own_score']-schema['means'][i])/schema['scales'][i]) if i is not None else 0.
    return dict(win_target=win, margin_target=float(margin), own_target=float(own),
                auxiliary_targets=np.asarray(auxiliary, np.float32),
                auxiliary_mask=np.asarray(mask, bool))


def reward(row, mode, game):
    if mode=='win' or game=='toy_battle': return row['win_target']
    if mode=='mixed': return .5*row['win_target']+.5*row['margin_target']
    if mode=='own': return .25*row['win_target']+.75*row['own_target']
    raise ValueError('Unknown training reward')


def teacher_distribution(actions, chosen, detail, objective):
    n = len(actions); target = np.zeros(n, np.float32)
    target[actions.index(chosen)] = 1.
    candidates = detail.get('candidates', [])
    usable = [c for c in candidates if c.get('samples', 0)>=4 and c.get('mean') is not None]
    if len(usable)>1:
        values = np.asarray([c['mean'] for c in usable], np.float64)
        temperature = .15 if objective=='win' else max(2., float(values.std()))
        probabilities = np.exp((values-values.max())/temperature); probabilities /= probabilities.sum()
        target *= .5
        for c, p in zip(usable, probabilities): target[actions.index(c['action'])] += .5*p
    return target


def read_result(path):
    path = Path(path); result = json.loads(path.read_text())
    if result['status']!='complete' or result['search_seconds']>result['budget']:
        raise ValueError('Only valid complete teacher games are accepted')
    trace_path = path.parent/result['trajectory']
    if hashlib.sha256(trace_path.read_bytes()).hexdigest()!=result['trajectory_sha256']:
        raise ValueError('Teacher trajectory checksum mismatch')
    with gzip.open(trace_path, 'rt') as f: trace = json.load(f)['trajectory']
    return result_labels(result), trace


def encode_experts(game, paths, base_spec, schema=None, anchor_scale=1., symbols=None):
    episodes = [read_result(p) for p in paths]
    if not episodes: raise ValueError('No teacher episodes')
    schema = schema or calibrate([r for r, _ in episodes])
    encoder = PublicTreeEncoder(game,symbols=symbols); rows = []
    for result, trace in episodes:
        base = base_policy(game, base_spec, result['seed'])
        seat = result['search_seat']; label = targets(result, seat, schema)
        for item in trace:
            if item['player']!=seat: continue
            legal = item['legal_actions']; detail = item['search']
            row = encode_record(game, encoder, base, item['observation'], legal, anchor_scale)
            row.update(label, teacher=teacher_distribution(legal, item['action'], detail, result['objective']),
                       action=legal.index(item['action']), seed=result['seed'], seat=seat,
                       teacher_rounds=detail['completed_rounds'],
                       teacher_weight=1. if detail['completed_rounds']>=4 else .25)
            rows.append(row)
    return dict(format='v173_encoded_experts_v1', game=game, rows=rows, schema=schema,
                base_spec=base_spec, anchor_scale=anchor_scale, encoding=encoder.metadata,
                episodes=len(episodes), sources=[str(Path(p).resolve()) for p in paths])


def initial_sources(game):
    refs = json.loads(Path('outputs/v172/retained_references_20261101.json').read_text())['games']
    spec = refs[game]
    roots = [Path(spec['validation_root'])]
    if game=='harmonies': roots.append(Path(refs['harmonies_score']['validation_root']))
    paths = []
    for root in roots:
        for p in (root/'games'/game/'matches').glob('*.json'):
            if p.name.endswith(('.progress.json', '.error.json')): continue
            r = json.loads(p.read_text())
            if r['budget']==900: paths.append(p)
    return sorted(paths), spec['rl_spec']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', choices=GAMES, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-teacher', type=Path)
    a = p.parse_args(); paths, spec = initial_sources(a.game)
    schema = None;symbols=None
    if a.extra_teacher:
        original = json.loads((a.output/'dataset.json').read_text())
        schema = original['schema']; spec = original['base_spec']; paths = []
        frozen=a.output/'vocabulary.json'
        if frozen.exists():symbols=json.loads(frozen.read_text())['symbols']
        for path in (a.extra_teacher/'games'/a.game/'matches').glob('*.json'):
            if path.name.endswith(('.progress.json','.error.json')): continue
            result = json.loads(path.read_text())
            if result['status']=='complete' and result['search_seconds']<=result['budget']:paths.append(path)
    packet = encode_experts(a.game, paths, spec, schema=schema, symbols=symbols)
    filename = 'extra_data.pkl.z' if a.extra_teacher else 'initial_data.pkl.z'
    sha = dump_packet(a.output/filename, packet)
    if a.extra_teacher:
        write_json(a.output/'extra_dataset.json',dict(path=str((a.output/filename).resolve()),sha256=sha,
            decisions=len(packet['rows']),episodes=packet['episodes'],sources=packet['sources']))
        print(json.dumps(dict(game=a.game,extra_decisions=len(packet['rows']),episodes=packet['episodes'])),flush=True)
        return
    write_json(a.output/'dataset.json', dict(game=a.game, path=str((a.output/'initial_data.pkl.z').resolve()),
        sha256=sha, decisions=len(packet['rows']), episodes=packet['episodes'], schema=packet['schema'],
        sources=packet['sources'], base_spec=spec))
    encoder=PublicTreeEncoder(a.game,symbols=symbols)
    write_json(a.output/'vocabulary.json',dict(game=a.game,sha256=encoder.vocabulary_sha256,
        symbols=[k for k,v in sorted(encoder.symbols.items(),key=lambda x:x[1])]))
    print(json.dumps(dict(game=a.game, decisions=len(packet['rows']), episodes=packet['episodes'])), flush=True)


if __name__=='__main__': main()
