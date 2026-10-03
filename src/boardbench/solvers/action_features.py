"""Public local relations, not heuristic scores or hidden future, for shared actors."""
from boardbench.environments.micro import EDGES, LINES as MICRO_LINES
from boardbench.environments.take_it_easy import TILES, LINES
from boardbench.environments.numerical import features

ACTION_FEATURE_VERSION = 'board_action_features_v1'
WIDTHS = {'micro_tiles': 12, 'take_it_easy': 16}


def action_features(obs):
    task, board = obs['task_id'], obs['board']
    rows = []
    if task == 'micro_tiles':
        for color in (obs['offers'] or [0, 0]):
            for cell in range(9):
                neighbors = [b if a==cell else a for a,b in EDGES if cell in (a,b)]
                groups = [neighbors]+[[i for i in line if i!=cell] for line in MICRO_LINES if cell in line]
                row = []
                for group in groups:
                    row += [sum(board[i]==color for i in group)/4,
                            sum(board[i]>=0 and board[i]!=color for i in group)/4,
                            sum(board[i]<0 for i in group)/4]
                row += [len(neighbors)/4, obs['placed_count']/9, float(len(set(obs['offers']))==1)]
                rows.append(row)
    elif task == 'take_it_easy':
        tile = TILES[obs['current_tile']] if obs['current_tile'] is not None else (0,0,0)
        for cell in range(19):
            row = []
            for axis in range(3):
                line = next(line['cells'] for line in LINES if line['axis']==axis and cell in line['cells'])
                others = [i for i in line if i!=cell]
                values = [TILES[board[i]][axis] for i in others if board[i]>=0]
                row += [values.count(tile[axis])/5,
                        sum(v!=tile[axis] for v in values)/5,
                        sum(board[i]<0 for i in others)/5, tile[axis]/9, len(line)/5]
            row += [obs['placed_count']/19]
            rows.append(row)
    else:
        raise ValueError('shared action features support Micro and Take It Easy only')
    assert all(len(row)==WIDTHS[task] for row in rows)
    return rows


def policy_features(obs, network='flat_mlp'):
    global_features = features(obs)
    if network == 'flat_mlp':
        return global_features
    if network != 'action_mlp':
        raise ValueError('unknown policy network')
    return global_features+tuple(v for row in action_features(obs) for v in row)
