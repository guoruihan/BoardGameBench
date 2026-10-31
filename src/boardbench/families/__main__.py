"""Small runnable entry point for the original two-player family."""
import argparse
import json
from pathlib import Path

from .play import run_match, write_json
from .registry import GAME_IDS, available_games, game_class


def specification(value):
    if value in ('random', 'heuristic'):
        return dict(kind=value)
    return json.loads(Path(value).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    match = commands.add_parser('match')
    match.add_argument('--game', required=True, choices=GAME_IDS)
    match.add_argument('--bot0', default='heuristic', help='random, heuristic or frozen bot JSON')
    match.add_argument('--bot1', default='random', help='random, heuristic or frozen bot JSON')
    match.add_argument('--seed', type=int, default=0)
    match.add_argument('--first-player', type=int, choices=(0, 1), default=0)
    match.add_argument('--max-seconds', type=float, default=300.)
    match.add_argument('--max-decisions', type=int, default=10000)
    match.add_argument('--output', type=Path)
    match.add_argument('--trajectory', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'list':
        available = available_games()
        print(json.dumps([dict(game_id=g, installed=g in available,
                   **({k:v for k,v in game_class(g).describe().items() if k != 'game_id'}
                      if g in available else {})) for g in GAME_IDS], indent=2))
        return
    result = run_match(args.game, [specification(args.bot0), specification(args.bot1)],
                       args.seed, args.first_player, args.max_decisions, args.max_seconds,
                       record=args.trajectory, validate=True)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'trajectory'}, ensure_ascii=False, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
