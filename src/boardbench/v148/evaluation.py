"""Reuse native batched complete-duo evaluation with fresh V148 seed namespaces."""
from boardbench.v147.training import Trainer as BaseTrainer

LAYOUT='duo_public_mcts_policy_iteration_v148'


class Trainer(BaseTrainer):
    VERSION=148
    LAYOUT=LAYOUT
    TRAINING_BASE=148000000
    DEVELOPMENT_BASE=248000000
    CONFIRMATION_BASE=348000000
