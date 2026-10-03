"""Tiny actual fitting job: estimate mean score from observed completed games."""
import sys

from boardbench.artifacts.store import read_json, write_json


def main():
    data = read_json(sys.argv[1])
    scores = data["scores"]
    if not scores:
        raise ValueError("cannot fit without observations")
    model = {"version": data["version"], "mean_score": sum(scores) / len(scores),
             "samples": len(scores)}
    write_json(sys.argv[2], model)
    print(f"Fitted mean_score={model['mean_score']} from {len(scores)} observations", flush=True)


if __name__ == "__main__":
    main()
