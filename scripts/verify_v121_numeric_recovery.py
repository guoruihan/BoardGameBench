"""Bounded mature-policy import/update gate; never modifies the parent campaign."""
import argparse
from pathlib import Path
import time

import torch

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v121.recovery import prepare_recovery
from boardbench.v121.registry import NUMERIC, prepare
from boardbench.v121.training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--stress-batches', type=int, default=32)
    args = parser.parse_args()
    if args.batches < 1 or args.stress_batches < args.batches:
        raise ValueError('positive batch counts, stress >= regular required')
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    initial_source = source_id()
    started = time.monotonic()
    prepare(root/'plan', tuple(NUMERIC), 7)
    recovery = prepare_recovery(args.parent_root, root, tuple(NUMERIC), (911,912,913))
    results = []
    for key, entry in recovery['entries'].items():
        tick = time.monotonic()
        trainer = Trainer(read_json(root/'plan/configs'/f'{key}.json'), 'cuda')
        try:
            trainer.import_numeric_checkpoint(entry['resume'], entry['parent_source'], entry['parent_sha256'])
            initial = trainer.steps
            out = root/key; out.mkdir()
            trainer.failure_directory = out
            trainer.save(out/'imported.resume.pt')
            rows = []
            count = args.stress_batches if key == 'E11_T16_F64_913' else args.batches
            for _ in range(count):
                row = trainer.ppo_batch(); rows.append(row)
                assert row['initial_ratio_max_error'] < 1e-10
            trainer.save(out/'verified.resume.pt')
            expected = {k:v.clone() for k,v in trainer.solver.model.state_dict().items()}
            trainer.load(out/'verified.resume.pt')
            assert all(torch.equal(v, trainer.solver.model.state_dict()[k]) for k,v in expected.items())
            update_seconds = sum(r['update_seconds'] for r in rows)
            collection_seconds = sum(r['environment_seconds'] + r['model_seconds'] for r in rows)
            result = {'run':key, 'parent_actions':initial, 'final_actions':trainer.steps,
                'batches':count, 'max_ratio_error':max(r['initial_ratio_max_error'] for r in rows),
                'all_losses_finite':all(torch.isfinite(torch.tensor(r['loss'])) for r in rows),
                'updates_added':trainer.updates-entry['parent_updates'],
                'actions_per_second':(trainer.steps-initial)/(update_seconds+collection_seconds),
                'wall_seconds':time.monotonic()-tick, 'rows':rows,
                'verified_resume_sha256':digest(out/'verified.resume.pt')}
            assert result['updates_added'] > 0 and result['all_losses_finite']
            write_json(out/'verification.json', result)
            results.append({k:v for k,v in result.items() if k != 'rows'})
            print(results[-1], flush=True)
        finally:
            trainer.close()
    if initial_source != source_id():
        raise RuntimeError('source changed during verification')
    write_json(root/'summary.json', {'source_id':initial_source, 'results':results,
        'gpu_reservation_seconds':time.monotonic()-started, 'diagnostic_only':True,
        'accounting':'all12 imports/initialization/updates/checkpoints included; not formal curve points'})


if __name__ == '__main__':
    main()
