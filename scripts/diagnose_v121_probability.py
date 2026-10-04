"""Read-only replay of a trusted local probability failure; run on a reserved GPU."""
import argparse
import json
import time

import numpy as np
import torch

from boardbench.v121.neural import ResearchSolver
from boardbench.v12.training import policy_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = torch.load(args.witness, map_location='cpu', weights_only=False)
    rows = data['rows']
    old = torch.tensor([r[3] for r in rows], dtype=torch.float64, device=args.device)
    masks = torch.tensor(np.stack([r[1] for r in rows]), device=args.device)
    actions = torch.tensor([r[2] for r in rows], device=args.device)
    print(json.dumps({'event': 'environment', 'torch': torch.__version__,
        'device': str(torch.cuda.get_device_name()) if args.device == 'cuda' else args.device,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'info': data['info']}), flush=True)
    for dtype in (torch.float32, torch.float64):
        model = ResearchSolver(data['config']['training_seed'], **data['config']['network'], device=args.device).model
        model.to(dtype=dtype)
        model.load_state_dict(data['model'], strict=True)
        x = torch.tensor(np.stack([r[0] for r in rows]), dtype=dtype, device=args.device)
        outputs = {}
        for size in (1, 2, 4, 8, 16, 32, 64, 128, 256, len(rows), len(rows)):
            start = time.monotonic()
            with torch.no_grad():
                lp = torch.cat([policy_distribution(model(x[i:i+size], masks[i:i+size])[0]).log_prob(actions[i:i+size])
                                for i in range(0, len(rows), size)])
            if args.device == 'cuda':
                torch.cuda.synchronize()
            delta = (lp - old).expm1().abs()
            row = {'event': 'replay', 'dtype': str(dtype), 'batch_size': size,
                   'seconds': time.monotonic()-start, 'vs_behavior_max_ratio_error': delta.max().item(),
                   'max_row': delta.argmax().item(), 'vs_batch1_max_ratio_error':
                   (lp-outputs.get(1, lp)).expm1().abs().max().item(),
                   'repeat_max_logp_error': (lp-outputs.get(size, lp)).abs().max().item()}
            outputs[size] = lp
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
