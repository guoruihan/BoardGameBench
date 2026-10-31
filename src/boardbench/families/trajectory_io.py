"""Lossless buffered archives and compact worker transport for family PPO.

Version 2 is standard concatenated gzip JSONL: one batch header followed by one
full episode per line. Compression happens in the CPU actors; the coordinator
writes already compressed bytes without rebuilding the observation object graph.
"""
import gzip
import json
from pathlib import Path
import time

import numpy as np


FORMAT = 'boardbench_full_trajectory_jsonl_v2'
TRAINING_FIELDS = ('x', 'prior', 'action', 'old_logp', 'value', 'advantage', 'return',
                   'legal_mask', 'player_id', 'policy_version', 'opponent_id')


def json_default(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    raise TypeError(type(value).__name__)


def encode_json(value):
    # dumps uses the C encoder; dump(iterencode) writes millions of scalar chunks
    # through TextIOWrapper into gzip for the same archive.
    return json.dumps(value, default=json_default, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_bytes(data);temporary.replace(path)


def write_compressed(path, data):
    atomic_bytes(path,gzip.compress(encode_json(data),compresslevel=1,mtime=0))


def compact_episode(episode):
    start=time.perf_counter()
    encoded=encode_json(episode)+b'\n'
    encoded_at=time.perf_counter()
    compressed=gzip.compress(encoded,compresslevel=1,mtime=0)
    rows=[]
    for original in episode['records']:
        row={k:original[k] for k in TRAINING_FIELDS}
        row['legal_action_count']=len(original['legal_actions'])
        rows.append(row)
    return dict(result=episode['result'],records=rows,archive_member=compressed,
        archive_profile=dict(serialization_seconds=encoded_at-start,
             compression_seconds=time.perf_counter()-encoded_at,
             uncompressed_bytes=len(encoded),compressed_bytes=len(compressed)))


def write_episode_batch(path, metadata, episodes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    header=dict(format=FORMAT,episode_count=len(episodes),metadata=metadata)
    with temporary.open('wb') as handle:
        handle.write(gzip.compress(encode_json(header)+b'\n',compresslevel=1,mtime=0))
        for episode in episodes:
            handle.write(episode['archive_member'])
    temporary.replace(path)


def read_episode_batch(path):
    """Read both old single-object .json.gz and new lossless .jsonl.gz archives."""
    if not str(path).endswith('.jsonl.gz'):
        with gzip.open(path,'rt') as handle:return json.load(handle)
    with gzip.open(path,'rt') as handle:
        header=json.loads(next(handle))
        if header.get('format')!=FORMAT:raise ValueError('Unknown trajectory format')
        episodes=[json.loads(line) for line in handle if line.strip()]
    if len(episodes)!=header['episode_count']:raise ValueError('Incomplete trajectory batch')
    return dict(header['metadata'],games=episodes)
