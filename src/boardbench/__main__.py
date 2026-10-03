import argparse
import json
from pathlib import Path
import sys

from boardbench.artifacts.store import read_json


def main():
    parser = argparse.ArgumentParser(description="BoardBench local experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--checkpoint", required=command == "evaluate")
    play = sub.add_parser("play", aliases=["serve"])
    play.add_argument("--config", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--trajectory", required=True)
    for p in (play, replay):
        p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "run":
            from boardbench.runner.engine import run
            result = run(read_json(args.config), args.out, checkpoint=args.checkpoint)
        elif args.command == "evaluate":
            from boardbench.runner.engine import evaluate
            result = evaluate(args.checkpoint, read_json(args.config), args.out)
        else:
            from boardbench.ui.server import serve
            serve(config=read_json(args.config) if args.command in ("play", "serve") else None,
                  trajectory=args.trajectory if args.command == "replay" else None, port=args.port)
            return 0
        print(json.dumps({"status": result["status"],
                          "summary": str(Path(args.out).resolve() / "summary.json"),
                          "checkpoints": [str(Path(args.out).resolve() / p) for p in result["checkpoints"]]},
                         ensure_ascii=False))
        return 1 if result["status"] == "failed" else 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"boardbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
