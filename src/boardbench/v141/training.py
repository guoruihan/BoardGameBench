"""Continue V140 in a new source/run while preserving all learning state."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v140.training import run


def continuation_config(parent, source):
    parent = Path(parent).resolve()
    if not (parent/'interrupted.json').exists() or (parent/'done.json').exists():
        raise ValueError('gracefully interrupted, unfinished parent required')
    config = read_json(parent/'config.json')
    provenance = dict(previous_trial=str(parent), previous_source_id=config['source_id'],
                      latest_sha256=read_json(parent/'latest.json')['sha256'],
                      best_sha256=read_json(parent/'best.json')['sha256'])
    return {**config, 'source_id': source, 'execution_version': 141, 'continuation': provenance}


def migrate(parent, out, config):
    """Only explicit execution provenance changes; never reset optimizer/budget."""
    parent, out = Path(parent).resolve(), Path(out)
    expected = continuation_config(parent, config['source_id'])
    if config != expected: raise ValueError('continuation changes protected training configuration')
    states = {}
    for name in ('latest', 'best'):
        path = parent/(name+'.pt'); meta = read_json(path.with_suffix('.json'))
        if digest(path) != config['continuation'][name+'_sha256'] or meta['sha256'] != digest(path):
            raise ValueError('parent checkpoint digest mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['version'] != 140 or state['config'] != read_json(parent/'config.json')
                or meta['source_id'] != config['continuation']['previous_source_id']):
            raise ValueError('parent source/config mismatch')
        state['config'] = deepcopy(config)
        states[name] = state
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', config)
    for name, state in states.items():
        path = out/(name+'.pt'); temporary = path.with_suffix('.partial.pt')
        torch.save(state, temporary); temporary.replace(path)
        write_json(path.with_suffix('.json'), dict(sha256=digest(path), batches=state['batches'],
                   actions=state['actions'], source_id=config['source_id']))
    write_json(out/'migration.json', dict(**config['continuation'], new_source_id=config['source_id'],
        batches=states['latest']['batches'], actions=states['latest']['actions'],
        elapsed_seconds=states['latest']['elapsed_seconds'], deadline_unix=config.get('deadline_unix'),
        preserved='weights, Adam, frozen/lagged opponents, RNG, counters, best selection and deadline',
        exact_legacy_random_draw_order=False, reason='model identity grouping changes draw ordering'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); c = read_json(a.config)
    if not a.resume: migrate(c['continuation']['previous_trial'], a.out, c)
    run(c, a.out, a.device, resume=True)
