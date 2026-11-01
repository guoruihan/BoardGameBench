"""Retain a frozen policy prior while shrinking learned actor corrections."""
import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch

from .guided_neural import load_model,save_model


def shrink_residual(model,scale):
    if not math.isfinite(scale) or not 0<=scale<=1:
        raise ValueError('Residual scale must lie in [0, 1]')
    result=deepcopy(model)
    with torch.no_grad():
        for layer in (result.core.actor[-1],result.feature_actor[-1]):
            layer.weight.mul_(scale);layer.bias.mul_(scale)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scale',type=float,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError('Keep source and variants immutable; choose a new output')
    model,encoder,metadata=load_model(a.source,a.game)
    source_hash=hashlib.sha256(a.source.read_bytes()).hexdigest()
    metadata=dict(metadata,residual_variant=dict(source=str(a.source.resolve()),sha256=source_hash,scale=a.scale))
    save_model(a.output,shrink_residual(model,a.scale),encoder,metadata)


if __name__=='__main__':main()
