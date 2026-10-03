"""Example policies. Runner never switches behavior based on these types."""
from .examples import RandomSolver, ReferenceSearch, CollaborationDemo
from .board import BoardSolver

SOLVERS = {"random": RandomSolver, "reference_search": ReferenceSearch,
           "collaboration_demo": CollaborationDemo, "board": BoardSolver}


def make_solver(config, seed):
    if config["id"] == "ppo":
        from .neural import NeuralSolver
        return NeuralSolver(seed=seed, **config.get("params", {}))
    return SOLVERS[config["id"]](seed=seed, **config.get("params", {}))
