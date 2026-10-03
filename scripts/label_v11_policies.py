"""Compact display-only policy labels; preserve complete training provenance."""
import argparse
from pathlib import Path

from boardbench.artifacts.store import digest, read_json, write_json


def display_label(entry):
    method = entry['method']
    if method not in ('rl', 'imitation'):
        return {'random': '随机', 'heuristic': '启发式', 'search': '搜索'}[method]
    family = 'PPO' if method == 'rl' else 'BC'
    initial = '初始' if entry['training']['episodes'] == 0 else ''
    return f"{family}{initial} · {entry['training_seed']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/v11')
    parser.add_argument('--out', default='runs/v11_policy_labels.json')
    args = parser.parse_args()
    changes = []
    for path in sorted(Path(args.root).glob('*/policies/*/manifest.json')):
        entry = read_json(path)
        protected = {key: digest(path.parent / entry[key])
                     for key in ('config_path', 'artifact_path')}
        before = dict(entry)
        entry['label'] = display_label(entry)
        write_json(path, entry)
        actual = read_json(path)
        assert {k: v for k, v in actual.items() if k != 'label'} == {
            k: v for k, v in before.items() if k != 'label'}
        assert protected == {key: digest(path.parent / entry[key]) for key in protected}
        changes.append({'path': str(path), 'before': before['label'], 'after': entry['label']})
    assert len(changes) == 39
    write_json(args.out, {'status': 'passed', 'only_display_label_changed': True, 'changes': changes})
    print(f'Updated {len(changes)} display labels; weights and provenance unchanged')


if __name__ == '__main__':
    main()
