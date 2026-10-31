"""Non-overlapping, exactly JSON-representable experiment seed partitions."""

PARTITIONS = {'cem_train': 1, 'cem_gate': 2, 'cem_confirmation': 3,
              'ppo_train': 4, 'ppo_gate': 5, 'ppo_confirmation': 6}


def experiment_seed(split, iteration=0, index=0):
    """Disjoint bit fields support 2**32 iterations and 2**16 games each.

    Every result is <2**53, so JSON/JavaScript round-trips preserve integers.
    Separate algorithms and train/gate/confirmation phases never share a seed.
    """
    if split not in PARTITIONS:
        raise ValueError('Unknown experiment split')
    if type(iteration) is not int or not 0 <= iteration < 2**32:
        raise ValueError('Iteration outside seed partition')
    if type(index) is not int or not 0 <= index < 2**16:
        raise ValueError('Game index outside seed partition')
    return (PARTITIONS[split] << 48) | (iteration << 16) | index
