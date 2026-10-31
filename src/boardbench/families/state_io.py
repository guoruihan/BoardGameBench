"""Fast copies of the family's builtin-container state and public JSON trees.

The bytes below are produced and consumed inside one call, never read from an
external pickle. Protocol 5 preserves integer keys, tuples, sets and shared
references used by private snapshots, unlike a JSON encode/decode shortcut.
Game state/observations/actions contain data, not custom copy hooks or models.
"""
import pickle


def copy_data(value):
    return pickle.loads(pickle.dumps(value, protocol=5))
