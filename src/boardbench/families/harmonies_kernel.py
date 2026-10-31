"""Equivalent public-board scoring: bitsets, with an explicitly pinned C++ option.

This is an execution backend for the existing candidate features, not a search
policy or a new game engine. It never receives the hidden supply or RNG.
"""
import argparse
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from boardbench.environments.harmonies import CARDS, NEIGHBORS, PATTERNS, STACKS, STACK_INDEX
from boardbench.environments.harmonies_duo import HarmoniesDuo

NEIGHBOR_BITS = tuple(sum(1 << j for j in neighbors) for neighbors in NEIGHBORS)
TOP = tuple(0 if not stack else 1 << ('gray', 'blue', 'brown', 'green', 'yellow', 'red').index(stack[-1])
            for stack in STACKS)
BONUS = (0., 0., 0., .25, .5, 0., .7, 1.8, 0., 0., 0., 1., 0., 0., 0.)
KINDS = ('choose_bundle', 'take_card', 'place_token', 'place_animal', 'end_turn')
PATTERN_ROWS = {}
for card, anchors in PATTERNS.items():
    rows = []
    for anchor, orientations in enumerate(anchors):
        for pattern in orientations.values():
            rows.append((anchor, tuple((cell, sum(1 << sid for sid in allowed),
                sum(1 << sid for sid, stack in enumerate(STACKS)
                    if any(STACKS[target][:len(stack)] == stack for target in allowed)))
                for cell, allowed in pattern)))
    PATTERN_ROWS[card] = tuple(rows)

_NATIVE = None


def neighbors(bits):
    result = 0
    while bits:
        bit = bits & -bits
        result |= NEIGHBOR_BITS[bit.bit_length()-1]
        bits ^= bit
    return result


@lru_cache(maxsize=8192)
def terrain(ids):
    """Return the reference integer terrain breakdown, including river diameter."""
    if _NATIVE is not None:
        return _NATIVE.terrain(ids)
    fields_mask = water = mountains_mask = 0
    trees = mountains = fields = buildings = river_length = 0
    for i, sid in enumerate(ids):
        trees += {8: 1, 9: 3, 10: 7}.get(sid, 0)
        if sid == 2:fields_mask |= 1 << i
        elif sid == 1:water |= 1 << i
        elif sid in (3, 4, 5):mountains_mask |= 1 << i
    for i, sid in enumerate(ids):
        if sid in (3, 4, 5) and NEIGHBOR_BITS[i] & mountains_mask:
            mountains += (1, 3, 7)[sid-3]
        elif sid in (12, 13, 14):
            colors = 0
            for j in NEIGHBORS[i]:colors |= TOP[ids[j]]
            buildings += 5 if colors.bit_count() >= 3 else 0
    while fields_mask:
        component = fields_mask & -fields_mask
        while True:
            expanded = component | (neighbors(component) & fields_mask)
            if expanded == component:break
            component = expanded
        fields += 5 if component.bit_count() >= 2 else 0
        fields_mask &= ~component
    starts = water
    while starts:
        frontier = starts & -starts;starts ^= frontier
        visited = frontier;length = 0
        while frontier:
            length += 1
            frontier = neighbors(frontier) & water & ~visited
            visited |= frontier
        river_length = max(river_length, length)
    river = (0, 0, 2, 5, 8, 11, 15)[river_length] if river_length <= 6 else 15+4*(river_length-6)
    return trees, mountains, fields, buildings, river, river_length


def board_score(player):
    scores = dict(zip(('trees', 'mountains', 'fields', 'buildings', 'river', 'river_length'),
                      terrain(tuple(c['stack_id'] for c in player['board']))))
    scores['animals'] = sum(CARDS[c['card_id']]['score_by_placed_count'][c['placed_count']]
                           for c in player['active_cards'] + player['completed_cards'])
    scores['total'] = sum(scores[k] for k in ('trees', 'mountains', 'fields', 'buildings', 'river', 'animals'))
    return scores


def landscape(ids, animals):
    return sum(terrain(ids)[:5]) + animals + sum(BONUS[sid] for sid in ids)


@lru_cache(maxsize=8192)
def potential(ids, occupied, card):
    best = 0.
    for anchor, pattern in PATTERN_ROWS[card]:
        if occupied & (1 << anchor):continue
        matched = 0
        for cell, allowed, reachable in pattern:
            bit = 1 << ids[cell]
            if allowed & bit:matched += 1
            elif occupied & (1 << cell) or not reachable & bit:break
        else:best = max(best, matched/len(pattern))
    return best


def features(observation, actions):
    if observation['rules_version'] != HarmoniesDuo.rules_version:
        raise ValueError('official duo required')
    player = observation['players'][observation['active_player']]
    ids = tuple(cell['stack_id'] for cell in player['board'])
    occupied = sum(1 << i for i, cell in enumerate(player['board']) if cell['animal'] is not None)
    animals = observation['score_breakdown'][observation['active_player']]['animals']
    encoded = []
    for action in actions:
        kind = KINDS.index(action['type'])
        if kind == 2:
            cell = action['cell_id'];target = STACK_INDEX[STACKS[ids[cell]]+(action['color'],)]
            encoded.append((kind, cell, target))
        elif kind == 1:encoded.append((kind, observation['animal_market'][action['market_slot']], 0))
        else:encoded.append((kind, 0, 0))
    if _NATIVE is not None and any(k in (1, 2) for k, _, _ in encoded):
        values = _NATIVE.values(ids, occupied, animals, encoded)
    else:
        baseline = landscape(ids, animals) if any(k == 2 for k, _, _ in encoded) else 0.
        values = []
        for kind, cell_or_card, target in encoded:
            if kind == 2:
                changed = ids[:cell_or_card] + (target,) + ids[cell_or_card+1:]
                values.append(landscape(changed, animals)-baseline)
            elif kind == 1:values.append(.2+potential(ids, occupied, cell_or_card))
            else:values.append(0.)
    result = []
    for action, (kind, _, _), value in zip(actions, encoded, values):
        if kind == 0:
            value = sum({'gray': .6, 'blue': 1., 'brown': .7, 'green': 1.3, 'yellow': .8, 'red': .5}[c]
                        for c in observation['token_market'][action['bundle_id']])
        elif kind == 3:
            card = next(c for c in player['active_cards'] if c['instance_id'] == action['instance_id'])
            track = CARDS[card['card_id']]['score_by_placed_count'];n = card['placed_count']
            value = track[n+1]-track[n]+1.
        elif kind == 4:value = -1.
        row = [float(value)] + [float(kind == k) for k in range(5)]
        row += [len(player['active_cards']), len(player['completed_cards']), player['turn_index']]
        result.append(row+[0.]*(24-len(row)))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tables():
    lines = ['static const uint32_t adjacency[23] = {'+','.join(map(str, NEIGHBOR_BITS))+'};',
             'static const int tops[15] = {'+','.join(map(str, TOP))+'};',
             'struct Requirement { int cell; uint16_t allowed, reachable; };',
             'struct Pattern { int anchor, begin, size; };']
    requirements = [];patterns = [];offsets = [0]
    for card in range(1, 33):
        for anchor, pattern in PATTERN_ROWS[card]:
            patterns.append((anchor, len(requirements), len(pattern)))
            requirements.extend(pattern)
        offsets.append(len(patterns))
    for name, ctype, rows in (('requirements', 'Requirement', requirements), ('patterns', 'Pattern', patterns)):
        lines.append('static const '+ctype+' '+name+'[] = {' +
                     ','.join('{'+','.join(map(str, row))+'}' for row in rows)+'};')
    lines.append('static const int offsets[33] = {'+','.join(map(str, offsets))+'};')
    return '\n'.join(lines)+'\n'


SOURCE = Path(__file__).with_name('native')/'harmonies_values.cpp'
LAYOUT = 'harmonies_public_values_v1'


class NativeKernel:
    def __init__(self, artifact):
        if (artifact['layout'] != LAYOUT or artifact['source_sha256'] != digest(SOURCE)
                or artifact['tables_sha256'] != hashlib.sha256(tables().encode()).hexdigest()
                or artifact['sha256'] != digest(artifact['path'])):
            raise ValueError('Harmonies native source/library/tables mismatch')
        self.lib = ctypes.CDLL(str(artifact['path']))
        self.short = ctypes.c_int16
        pointer = ctypes.POINTER(self.short)
        self.score = self.lib.harmonies_terrain
        self.score.argtypes = [pointer, pointer];self.score.restype = ctypes.c_int
        self.evaluate = self.lib.harmonies_values
        self.evaluate.argtypes = [pointer, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                                  pointer, ctypes.POINTER(ctypes.c_double)]
        self.evaluate.restype = ctypes.c_int

    def packed(self, ids):
        if len(ids) != 23 or any(type(s) is not int or not 0 <= s < 15 for s in ids):
            raise ValueError('Expected 23 stack IDs in [0,15)')
        return (self.short*23)(*ids)

    def terrain(self, ids):
        packed = self.packed(ids);out = (self.short*6)()
        if self.score(packed, out):raise ValueError('Native terrain rejected input')
        return tuple(out)

    def values(self, ids, occupied, animals, encoded):
        if not 0 <= occupied < (1 << 23) or not 0 <= animals <= 10000 or len(encoded) > 256:
            raise ValueError('Invalid public feature input')
        packed = self.packed(ids);flat = [v for row in encoded for v in row]
        if len(flat) != len(encoded)*3 or any(type(v) is not int or not -32768 <= v < 32768 for v in flat):
            raise ValueError('Invalid candidate array')
        candidates = (self.short*len(flat))(*flat);out = (ctypes.c_double*len(encoded))()
        if self.evaluate(packed, occupied, animals, len(encoded), candidates, out):
            raise ValueError('Native features rejected input')
        return list(out)


def configure(artifact=None):
    """Explicit per-process backend; used by spawned actors and their learner."""
    global _NATIVE
    _NATIVE = NativeKernel(artifact) if artifact is not None else None
    terrain.cache_clear();potential.cache_clear()


def build(directory):
    directory = Path(directory).resolve();directory.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which('g++')
    if not compiler:raise RuntimeError('Explicit native build requires g++; Python backend remains available')
    (directory/'tables.h').write_text(tables())
    library = directory/'harmonies_values.so'
    command = [compiler, '-O3', '-std=c++17', '-shared', '-fPIC', '-fno-fast-math',
               '-I', str(directory), str(SOURCE), '-o', str(library)]
    start = time.monotonic();subprocess.run(command, check=True, timeout=120)
    artifact = dict(path=str(library), sha256=digest(library), source_sha256=digest(SOURCE),
        tables_sha256=digest(directory/'tables.h'), layout=LAYOUT, compiler_command=command,
        compiler_version=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
        build_seconds=time.monotonic()-start)
    (directory/'artifact.json').write_text(json.dumps(artifact, indent=2)+'\n')
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--build', required=True)
    print(json.dumps(build(parser.parse_args().build)))
