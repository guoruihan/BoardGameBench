"""Pinned C++ public-feature encoding. Build explicitly; never silently fall back."""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from boardbench.artifacts.store import digest, write_json
from boardbench.environments.harmonies import CARDS, COLORS
from boardbench.v12.encoding import CARD_VECTORS, MAX_CUBES

WIDTH, FEATURES = 161, 1346
CPP_SOURCE = Path(__file__).with_name('native')/'encode.cpp'


def pack(observations):
    rows = []
    for obs in observations:
        active = obs['active_cards']
        owned = {c['card_id']: c['placed_count'] for c in active + obs['completed_cards']}
        remaining = set(obs['remaining_cards'])
        row = [c['stack_id'] for c in obs['board']] + [c['animal'] or 0 for c in obs['board']]
        row += [(bundle or []).count(color) for bundle in obs['token_market'] for color in COLORS]
        for field in ('pending_tokens', 'remaining_tokens', 'discarded_tokens'):
            row += [obs[field][color] for color in COLORS]
        row += [c or 0 for c in obs['animal_market']]
        row += [c['card_id'] for c in active] + [0] * (4-len(active))
        row += [c['placed_count'] for c in active] + [0] * (4-len(active))
        row += [owned.get(c, -1) for c in range(1, 33)]
        row += [int(c in remaining) for c in range(1, 33)]
        row += [obs['turn_index'], obs['bundle_taken'], obs['card_taken'], obs['terminated']]
        if len(row) != WIDTH:
            raise ValueError('public state layout changed')
        rows.append(row)
    return np.asarray(rows, dtype=np.int16).reshape(len(rows), WIDTH)


class NativeEncoder:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.short, self.single = ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_float)
        self.fn = self.lib.harmonies_encode
        self.fn.argtypes = [ctypes.c_size_t, self.short, self.single, self.short, ctypes.c_int, self.single]
        self.fn.restype = ctypes.c_int
        self.cards = np.asarray([(0.,)*83] + [CARD_VECTORS[i] for i in range(1, 33)], dtype=np.float32)
        if self.cards.shape != (33, 83):
            raise ValueError('card encoding changed')
        self.cubes = np.asarray([0] + [CARDS[i]['cube_count'] for i in range(1, 33)], dtype=np.int16)
        self.card_pointer = self.cards.ctypes.data_as(self.single)
        self.cube_pointer = self.cubes.ctypes.data_as(self.short)

    @classmethod
    def from_artifact(cls, artifact):
        if (artifact['cpp_sha256'] != digest(CPP_SOURCE) or
                artifact['sha256'] != digest(artifact['path']) or artifact['layout'] != 'public161_cards1346_v1'):
            raise ValueError('native library/source/layout mismatch')
        return cls(artifact['path'])

    def encode(self, states):
        if states.dtype != np.int16 or states.ndim != 2 or states.shape[1] != WIDTH or not states.flags.c_contiguous:
            raise ValueError('contiguous public state array required')
        out = np.empty((len(states), FEATURES), dtype=np.float32)
        code = self.fn(len(states), states.ctypes.data_as(self.short), self.card_pointer,
                       self.cube_pointer, MAX_CUBES, out.ctypes.data_as(self.single))
        if code:
            raise ValueError(f'native encoder rejected state: {code}')
        return out


def build(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ required for explicit native build; no automatic install')
    library = out/'harmonies_encode.so'
    command = [compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(CPP_SOURCE), '-o', str(library)]
    try:
        subprocess.run(command, check=True, timeout=120)
        artifact = {'path': str(library), 'sha256': digest(library), 'cpp_sha256': digest(CPP_SOURCE),
                    'layout': 'public161_cards1346_v1', 'compiler_command': command,
                    'compiler_version': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                    'host': os.uname().nodename, 'build_seconds': time.monotonic()-start}
        write_json(out/'artifact.json', artifact)
        return artifact
    except BaseException as exc:
        write_json(out/'failed.json', {'error': repr(exc), 'seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', required=True)
    print(build(parser.parse_args().out))
