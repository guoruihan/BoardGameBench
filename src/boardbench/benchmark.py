"""Validation-only selection, portable policy export and frozen Runner tests."""
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import os
import multiprocessing
from pathlib import Path
import platform
import shutil
import statistics
import time
from types import SimpleNamespace

from boardbench.artifacts.store import (ArtifactStore, digest, read_json, save_checkpoint,
                                       source_id, write_json)
from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.runner.config import normalize
from boardbench.runner.context import empty_resources
from boardbench.runner.engine import evaluate
from boardbench.solvers.board import BoardSolver
from boardbench.provenance import admit_sources, training_candidates


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*q
    low = int(position)
    return values[low]+(values[min(low+1, len(values)-1)]-values[low])*(position-low)


def metrics(episodes, times, loading=()):
    scores = [e["score"] for e in episodes if e["status"] == "completed"]
    components = {}
    for key in ('terrain','animals','animal_placements','completed_cards'):
        values = [e['components'][key] for e in episodes if e['status']=='completed' and key in e.get('components',{})]
        if values:
            components[key] = statistics.mean(values)
    wall = [e['resources']['wall_seconds'] for e in episodes if 'wall_seconds' in e['resources']]
    return {"episodes": len(episodes), "completed": len(scores),
            "completion_rate": len(scores)/len(episodes) if episodes else 0.,
            "score_mean": statistics.mean(scores) if scores else None,
            "score_std": statistics.pstdev(scores) if scores else None,
            "failed": sum(e["status"] == "failed" for e in episodes),
            "truncated": sum(e["status"] == "truncated" for e in episodes),
            "decision_p50_seconds": percentile(times, .5), "decision_p95_seconds": percentile(times, .95),
            "mean_game_decision_seconds": statistics.mean([e["resources"]["decision_seconds"] for e in episodes]) if episodes else None,
            "mean_simulation_steps": statistics.mean([e["resources"]["simulation_steps"] for e in episodes]) if episodes else None,
            "mean_model_load_seconds": statistics.mean(loading) if loading else None,
            'mean_episode_wall_seconds': statistics.mean(wall) if wall else None,
            'score_components':components,
            "hardware": platform.node(), "device": "cpu", "threads": 2,
            "score_policy": "only completed scores; failures/truncations retained separately"}


def candidates(training):
    items = [{"id": "random", "method": "random", "params": {"method": "random"}},
             {"id": "heuristic", "method": "heuristic", "params": {"method": "heuristic"}}]
    for width, depth in ((2, 2), (4, 3)):
        items.append({"id": f"search_w{width}_d{depth}", "method": "search",
                      "params": {"method": "search", "width": width, "rollouts": 2,
                                 "depth": depth, "max_simulation_steps": width*2*depth}})
    items.extend(training_candidates(training))
    return items


def build_solver(task, candidate):
    if candidate["method"] in ("rl", "imitation"):
        from boardbench.solvers.neural import NeuralSolver
        if digest(candidate["weights"]) != candidate["weights_sha256"]:
            raise ValueError("training weights changed since candidate definition")
        solver = NeuralSolver(90210, task, hidden=candidate.get("hidden", 128), device="cpu",
                              network=candidate.get('network','flat_mlp'))
        solver.load_weights(candidate["weights"])
        return solver
    return BoardSolver(90210, **candidate["params"])


def validation_episode(arguments):
    task,candidate,seed = arguments
    started=time.perf_counter()
    solver=build_solver(task,candidate)
    loading=time.perf_counter()-started
    env=TASKS[task].factory()
    obs,_=env.reset(seed)
    simulation_steps=[0]
    times=[]
    def count(): simulation_steps[0]+=1
    ctx=SimpleNamespace(task_spec=TASKS[task].spec,reference_simulator=SearchAccess(env.fork,count))
    status,score,error,animal_placements='truncated',None,None,0
    rollouts=reached=0
    try:
        for _ in range(256):
            policy_obs=deepcopy(obs)
            tick=time.perf_counter()
            action=solver.decide(policy_obs,ctx)
            times.append(time.perf_counter()-tick)
            debug=getattr(solver,'last_search',{})
            rollouts+=debug.get('rollouts',0)
            reached+=debug.get('horizon_completed_rollouts',0)
            obs=env.step(action).observation
            animal_placements+=action.get('type')=='place_animal'
            if obs['terminated']:
                status,score='completed',obs['score']
                break
    except Exception as exc:
        status,error='failed',f'{type(exc).__name__}: {exc}'
    record={'seed':seed,'status':status,'score':score,'error':error,
            'resources':{'simulation_steps':simulation_steps[0],'decision_seconds':sum(times)},
            'search_horizon':{'rollouts':rollouts,'reached':reached}}
    if task=='harmonies' and status=='completed':
        record['components']={'terrain':score-obs['score_breakdown']['animals'],
                              'animals':obs['score_breakdown']['animals'],
                              'animal_placements':animal_placements,'completed_cards':len(obs['completed_cards'])}
    return record,times,loading


def validation_worker(expected_source):
    import torch
    torch.set_num_threads(2)
    if source_id()!=expected_source:
        raise ValueError('validation worker source changed after freeze')


def validate(task, training, output, extra_training=None, search_candidates=None, retain_runs=False, workers=1):
    import torch
    torch.set_num_threads(2)
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer')
    splits = read_json(Path(training)/"splits.json")
    definitions = candidates(training)
    if search_candidates is not None:
        if not search_candidates or any(c['method']!='search' for c in search_candidates):
            raise ValueError('search candidates must be a nonempty search-only list')
        definitions = [c for c in definitions if c['method']!='search']+deepcopy(search_candidates)
    extras = ([extra_training] if isinstance(extra_training, (str, Path))
              else list(extra_training or []))
    for extra in extras:
        definitions.extend(training_candidates(extra))
    # Reject invalid provenance before constructing a solver or creating outputs.
    admit_sources(task, definitions, splits)
    definitions = list({c['id']: c for c in definitions}.values())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    from boardbench.artifacts.store import source_files, source_root
    for relative in source_files():
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root()/relative, destination)
    validation_code_version = source_id(out / "source")
    write_json(out / "splits.json", splits)
    summaries = {}
    pool = (ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                initializer=validation_worker,initargs=(validation_code_version,)) if workers>1 else None)
    try:
        for candidate in definitions:
            times, episodes, loading = [], [], []
            records = out / (candidate["id"]+".jsonl")
            jobs=((task,candidate,seed) for seed in splits['validation'])
            results=pool.map(validation_episode,jobs) if pool else map(validation_episode,jobs)
            with records.open('x') as stream:
                for record,elapsed,load in results:
                    episodes.append(record)
                    times.extend(elapsed)
                    loading.append(load)
                    stream.write(json.dumps(record)+'\n')
                    stream.flush()
            summaries[candidate['id']]={**metrics(episodes,times,loading),'parallel_workers':workers}
            write_json(out/'summaries.json',summaries)
            print(json.dumps({'candidate':candidate['id'],**summaries[candidate['id']]}),flush=True)
    finally:
        if pool:
            pool.shutdown()
    selected = []
    for method in ("random", "heuristic", "search", "rl", "imitation"):
        pool = [c for c in definitions if c["method"] == method
                and not c.get('untrained', False) and summaries[c["id"]]["completed"] == len(splits["validation"])]
        if not pool:
            if method == 'imitation':
                continue
            raise RuntimeError(f"no fully evaluated candidate for {method}")
        chosen = max(pool, key=lambda c: (summaries[c["id"]]["score_mean"], -summaries[c["id"]]["decision_p95_seconds"]))
        selected.append(chosen)
        if retain_runs and method in ('rl','imitation'):
            selected.pop()
            expected_runs={c['origin']['run_id'] for c in definitions if c['method']==method and not c['untrained']}
            if expected_runs!={c['origin']['run_id'] for c in pool}:
                raise RuntimeError('a training repetition has no fully evaluated checkpoint')
            for run_id in sorted(expected_runs):
                run_pool=[c for c in pool if c['origin']['run_id']==run_id]
                selected.append(max(run_pool,key=lambda c:(summaries[c['id']]['score_mean'],
                                                            -summaries[c['id']]['decision_p95_seconds'])))
    selected_runs = {c['origin']['run_id'] for c in selected if c['method'] in ('rl','imitation')}
    selected += [c for c in definitions if c.get('untrained') and c['origin']['run_id'] in selected_runs]
    if any(summaries[c['id']]['completed']!=len(splits['validation']) for c in selected):
        raise RuntimeError('selected control/candidate did not complete validation')
    selection = {"task_id": task, "rules_version": TASKS[task].spec["version"],
                 "source_id": validation_code_version, "selection_split": "validation", "test_used": False,
                 "selected": selected, "candidates": definitions, "summaries": summaries, "splits": splits,
                 "training_config": read_json(Path(training)/"config.json")}
    write_json(out / "selection.json", selection)
    return selection


def export_policy(task, candidate, validation_summary, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    solver = build_solver(task, candidate)
    learned = candidate['method'] in ('rl','imitation')
    params = ({"task_id": task, "hidden": solver.hidden, "device": "cpu", 'network': solver.network} if learned
              else candidate["params"])
    config = normalize({"task": {"id": task, "version": TASKS[task].spec["version"]},
                        "solver": {"id": "ppo" if learned else "board", "params": params},
                        "episodes": 1, "seed": 90210, "model": {"max_calls": 0},
                        "limits": {"max_action_requests": 256, "max_actions_per_episode": 256, "max_wall_seconds": 120},
                        "capabilities": {"reference_simulator": candidate["method"] == "search"}})
    store = ArtifactStore(root / "export", config)
    try:
        checkpoint = save_checkpoint(store, solver, config, {"completed_episodes": 0, "started_episodes": 0},
                                     empty_resources(), store.root / "workspace")
    finally:
        store.close()
    artifact = checkpoint / "solver" / ("weights.pt" if learned else "board_solver.json")
    write_json(root / "config.json", config)
    write_json(root / "validation.json", validation_summary)
    training = candidate.get("training", {})
    manifest = {"policy_id": candidate["id"], "environment_id": task,
                "rules_version": TASKS[task].spec["version"], "method": candidate["method"],
                "label": (f"{candidate['method']} · seed {training['training_seed']} · {training['episodes']} episodes"
                          if training else candidate['id']), "code_version": source_id(),
                "training_code_version": training.get("code_version"),
                "config_path": "config.json", "config_sha256": digest(root / "config.json"),
                "artifact_path": str(artifact.relative_to(root.resolve())), "artifact_sha256": digest(artifact),
                "checkpoint_path": str(checkpoint.relative_to(root.resolve())), "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
                "training_steps": training.get("environment_steps"), "training_seed": training.get("training_seed"),
                "training": training or None, "origin": candidate.get('origin'), "selection_split": "validation",
                "validation_summary_path": "validation.json", "validation_mean": validation_summary["score_mean"],
                "inference_device": "cpu", "difficulty": None}
    write_json(root / "manifest.json", manifest)
    return checkpoint


def frozen_shard(arguments):
    checkpoint,evaluation,run_dir=arguments
    return evaluate(checkpoint,evaluation,run_dir)


def frozen_test(selection_path, output, policy_directory, workers=1, shard_size=32):
    import torch
    torch.set_num_threads(2)
    selection = read_json(selection_path)
    if selection["selection_split"] != "validation" or selection["test_used"]:
        raise ValueError("expected frozen validation-only selection")
    if selection['source_id'] != source_id():
        raise ValueError('runtime source differs from frozen validation; use its source snapshot')
    if type(workers) is not int or workers<1 or type(shard_size) is not int or shard_size<1:
        raise ValueError('positive workers and shard_size required')
    seeds=selection['splits']['test']
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('final test seeds must be nonempty and unique')
    if 'candidates' in selection:
        admit_sources(selection['task_id'],selection['selected'],selection['splits'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_selection.json", selection)
    write_json(out / "selection_provenance.json", {"input_sha256": digest(selection_path), "runtime_source_id": source_id()})
    results = {}
    for candidate in selection["selected"]:
        cid = candidate["id"]
        checkpoint = export_policy(selection["task_id"], candidate, selection["summaries"][cid], Path(policy_directory)/cid)
        evaluation = {"episode_seeds": selection["splits"]["test"],
                      "per_episode_limits": {"max_action_requests": 256, "max_wall_seconds": 120, "max_model_calls": 0}}
        write_json(out / (cid+"_eval_config.json"), evaluation)
        if workers==1:
            run_dirs=[out/cid]
            summaries=[evaluate(checkpoint,evaluation,run_dirs[0])]
        else:
            run_dirs=[out/cid/f'shard_{i//shard_size:03d}' for i in range(0,len(seeds),shard_size)]
            jobs=[(checkpoint,{**evaluation,'episode_seeds':seeds[i:i+shard_size]},run_dir)
                  for i,run_dir in zip(range(0,len(seeds),shard_size),run_dirs)]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=validation_worker,initargs=(source_id(),)) as pool:
                summaries=list(pool.map(frozen_shard,jobs))
            write_json(out/cid/'aggregation.json',{'raw_runs':[p.name for p in run_dirs],
                       'episode_seeds':seeds,'rule':'disjoint shards; original Runner logs retained unchanged'})
        episodes,times,loads,illegal=[],[],[],0
        for run_dir in run_dirs:
            part=[json.loads(line) for line in (run_dir/'episodes.jsonl').read_text().splitlines()]
            pure_per_episode,components,placements={},{},{}
            with (run_dir / "events.jsonl").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["kind"] == "decision_end":
                        pure = event["solver_seconds"]
                        times.append(pure)
                        ep = event["episode"]
                        pure_per_episode[ep] = pure_per_episode.get(ep, 0.)+pure
                    if event["kind"] == "hook_end" and event.get("hook") == "load":
                        loads.append(event["elapsed_seconds"])
                    illegal += event["kind"] == "action_error"
                    if event['kind']=='transition' and selection['task_id']=='harmonies':
                        ep=event['episode']
                        placements[ep]=placements.get(ep,0)+(event['action'].get('type')=='place_animal')
                        obs=event['result']['observation']
                        if obs['terminated']:
                            components[ep]={'terrain':obs['score']-obs['score_breakdown']['animals'],
                                            'animals':obs['score_breakdown']['animals'],
                                            'animal_placements':placements[ep],
                                            'completed_cards':len(obs['completed_cards'])}
            for episode in part:
                episode['resources']['decision_seconds']=pure_per_episode.get(episode['episode'],0.)
                episode['components']=components.get(episode['episode'],{})
            episodes.extend(part)
        if [e['seed'] for e in episodes]!=seeds:
            raise ValueError('sharded evaluation seed coverage/order mismatch')
        results[cid] = {**metrics(episodes, times, loads), "illegal_actions": illegal,
                        "runner_status": 'completed' if all(s['status']=='completed' for s in summaries) else 'failed',
                        'raw_runs':[str(p.relative_to(out)) for p in run_dirs], 'parallel_workers':workers,
                        "historical_training": candidate.get("training")}
        write_json(out / "results.json", results)
        print(json.dumps({"policy": cid, **results[cid]}), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--task", required=True, choices=("micro_tiles", "take_it_easy", "harmonies"))
    val.add_argument("--training", required=True)
    val.add_argument("--extra-training", action='append')
    val.add_argument('--search-config')
    val.add_argument('--retain-runs',action='store_true')
    val.add_argument('--workers',type=int,default=1)
    test = sub.add_parser("test")
    test.add_argument("--selection", required=True)
    test.add_argument("--policies", required=True)
    test.add_argument('--workers',type=int,default=1)
    test.add_argument('--shard-size',type=int,default=32)
    for item in (val, test):
        item.add_argument("--out", required=True)
    args = p.parse_args()
    if args.phase == "validate":
        searches=read_json(args.search_config)['search_candidates'] if args.search_config else None
        validate(args.task, args.training, args.out, args.extra_training, searches, args.retain_runs, args.workers)
    else:
        frozen_test(args.selection, args.out, args.policies, args.workers, args.shard_size)


if __name__ == "__main__":
    main()
