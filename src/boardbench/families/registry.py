"""The requested family; unavailable implementations are explicit, never aliases."""
import importlib
import importlib.util

GAME_IDS = (
    'cascadia', 'calico', 'patchwork', 'harmonies', 'century_spice_road',
    'jaipur', 'splendor', 'gizmos', 'machi_koro', 'clever', 'five_tribes',
    'century_a_new_world', 'stone_age', 'targi', 'dominion', 'el_dorado', 'toy_battle',
)


def game_class(game_id):
    if game_id not in GAME_IDS:
        raise ValueError(f'Unknown original game: {game_id}')
    name = f'boardbench.families.games.{game_id}'
    if importlib.util.find_spec(name) is None:
        raise NotImplementedError(f'{game_id}: no verified family implementation installed')
    return importlib.import_module(name).GAME_CLASS


def available_games():
    # Availability describes installed code, not a rule-completeness certification.
    return [g for g in GAME_IDS
            if importlib.util.find_spec(f'boardbench.families.games.{g}') is not None]
