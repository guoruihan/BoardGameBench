from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
import uuid

from boardbench.contracts import InvalidAction, derive_seed
from boardbench.environments import get_task, TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.artifacts.store import read_json, write_json
from .policies import PolicyRegistry


class PlaySession:
    def __init__(self, config):
        task_config = config.get("task", {"id": "risk_collect", "version": "1"})
        self.config = deepcopy(config)
        self.task = task_config
        self.env = get_task(self.task).factory()
        self.registry = None
        self.save_directory = Path(config.get("save_directory", "runs/ui_saves"))
        self._setup_policies()
        self.base_seed = config.get("seed", 123)
        self.game = 0
        self.new_game()

    def _setup_policies(self):
        if self.registry:
            self.registry.close()
        if self.task["id"] == "risk_collect":
            self.registry, self.policy_id = None, None
        else:
            directory = self.config.get("policy_directory")
            if self.task["id"] != self.config.get("task", {}).get("id"):
                directory = str(Path(self.config.get("policy_root", "outputs/v1")) / self.task["id"] / "policies")
            self.registry = PolicyRegistry(self.task, directory)
            self.policy_id = next(iter(self.registry.entries))

    def new_game(self, seed=None, task_id=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be a nonnegative 64-bit integer")
        if task_id is not None and task_id != self.task["id"]:
            if task_id not in TASKS:
                raise ValueError("unknown task")
            self.task = {"id": task_id, "version": TASKS[task_id].spec["version"]}
            self.env = get_task(self.task).factory()
            self._setup_policies()
        self.game += 1
        self.seed = derive_seed(self.base_seed, "play", self.game) if seed is None else seed
        self.observation, self.info = self.env.reset(self.seed)
        self.last_hint, self.challenge = None, None
        self.frames = [{"observation": deepcopy(self.observation), "info": deepcopy(self.info)}]
        return self.state()

    def step(self, action):
        result = self.env.step(action)
        self.observation, self.info = result.observation, result.info
        self.last_hint = None
        self.frames.append({"observation": deepcopy(self.observation), "info": deepcopy(self.info), "action": deepcopy(action)})
        return result

    def hint(self, policy_id=None):
        if self.registry is None:
            raise ValueError("policy controls are available for V1 games")
        if self.observation["terminated"]:
            raise ValueError("game finished")
        chosen = policy_id or self.policy_id
        solver, load_seconds = self.registry.get(chosen)
        self.policy_id = chosen
        count = [0]
        def tick():
            count[0] += 1
        context = SimpleNamespace(task_spec=get_task(self.task).spec,
                                  reference_simulator=SearchAccess(self.env.fork, tick))
        policy_observation = deepcopy(self.observation)
        started = time.perf_counter()
        action = solver.decide(policy_observation, context)
        elapsed = time.perf_counter()-started
        if action not in self.observation["legal_actions"]:
            raise ValueError("policy returned an illegal action")
        self.last_hint = {"action": action, "policy_id": chosen, "decision_seconds": elapsed,
                          "load_seconds": load_seconds, "simulation_steps": count[0]}
        return self.last_hint

    def save_game(self):
        if not hasattr(self.env, "save_state"):
            raise ValueError("full game saves are available for V1 games")
        save_id = uuid.uuid4().hex
        write_json(self.save_directory / (save_id+".json"), {"task": self.task,
                   "snapshot": self.env.save_state(), "seed": self.seed, "frames": self.frames,
                   "info": self.info})
        return save_id

    def load_game(self, save_id):
        if not isinstance(save_id, str) or not re.fullmatch(r"[0-9a-f]{32}", save_id):
            raise ValueError("invalid save ID")
        saved = read_json(self.save_directory / (save_id+".json"))
        task = saved["task"]
        env = get_task(task).factory()
        observation = env.load_state(saved["snapshot"])
        self.task, self.env, self.observation = task, env, observation
        self.seed, self.frames, self.info = saved["seed"], saved["frames"], saved["info"]
        self.last_hint, self.challenge = None, None
        self._setup_policies()

    def run_challenge(self, policy_id=None):
        if not self.observation["terminated"]:
            raise ValueError("finish the human game before revealing the comparison")
        if self.registry is None:
            raise ValueError("challenge requires a V1 game")
        chosen = policy_id or self.policy_id
        solver, load_seconds = self.registry.get(chosen, fresh=True)
        env = get_task(self.task).factory()
        obs, info = env.reset(self.seed)
        frames = [{"observation": deepcopy(obs), "info": info}]
        count = [0]
        def tick():
            count[0] += 1
        ctx = SimpleNamespace(task_spec=get_task(self.task).spec, reference_simulator=SearchAccess(env.fork, tick))
        total = 0.
        for _ in range(256):
            if obs["terminated"]:
                break
            policy_observation = deepcopy(obs)
            start = time.perf_counter()
            action = solver.decide(policy_observation, ctx)
            total += time.perf_counter()-start
            result = env.step(action)
            obs = result.observation
            frames.append({"observation": deepcopy(obs), "info": result.info, "action": action})
        if not obs["terminated"]:
            raise ValueError("challenge exceeded action bound")
        self.challenge = {"policy_id": chosen, "human_score": self.observation["score"],
                          "bot_score": obs["score"], "frames": frames, "decision_seconds": total,
                          "load_seconds": load_seconds, "simulation_steps": count[0]}
        return self.challenge

    def export_replay(self):
        replay_id = uuid.uuid4().hex
        self.save_directory.mkdir(parents=True, exist_ok=True)
        path = self.save_directory / (replay_id+".jsonl")
        records = [{"kind": "run_start", "task": self.task},
                   {"kind": "episode_start", "episode": 1, "observation": self.frames[0]["observation"]}]
        records += [{"kind": "transition", "episode": 1, "step": i, "action": frame["action"],
                     "result": {"observation": frame["observation"], "info": frame["info"]}}
                    for i, frame in enumerate(self.frames[1:], 1)]
        with path.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record)+"\n")
        return str(path)

    def state(self):
        return {"mode": "play", "episode": self.game, "observation": self.observation,
                "info": self.info, "seed": str(self.seed), "task": self.task,
                "task_spec": get_task(self.task).spec, "hint": self.last_hint,
                "policies": self.registry.list() if self.registry else [], "policy_id": self.policy_id,
                "challenge": self.challenge if self.observation["terminated"] else None,
                "human_frames": self.frames if self.observation["terminated"] else None}


class Replay:
    def __init__(self, trajectory):
        self.episodes = {}
        config = Path(trajectory).with_name("config.json")
        self.task = read_json(config).get("task") if config.is_file() else None
        with Path(trajectory).open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                episode = event.get("episode")
                if event["kind"] == "run_start":
                    self.task = event.get("task", self.task)
                elif event["kind"] == "episode_start":
                    self.episodes[episode] = [{"observation": event["observation"], "info": {}}]
                elif event["kind"] == "transition":
                    self.episodes[episode].append({"observation": event["result"]["observation"],
                                                   "info": event["result"]["info"], "action": event["action"]})
                elif event["kind"] == "episode_end" and episode in self.episodes:
                    self.episodes[episode][-1]["episode_result"] = event["result"]
        if not self.episodes:
            raise ValueError("trajectory contains no recorded episodes")
        if self.task is None:
            raise ValueError("replay requires explicit task metadata or sibling config.json")

    def state(self, episode=None, step=0):
        episode = min(self.episodes) if episode is None else episode
        if episode not in self.episodes:
            raise ValueError("unknown replay episode")
        frames = self.episodes[episode]
        if not 0 <= step < len(frames):
            raise ValueError("replay step out of range")
        return {"mode": "replay", "episode": episode, "episodes": list(self.episodes),
                "task": self.task, "task_spec": get_task(self.task).spec,
                "step": step, "max_step": len(frames) - 1, **frames[step]}


def make_server(*, config=None, trajectory=None, port=8765):
    session = Replay(trajectory) if trajectory is not None else PlaySession(config or {})
    page = Path(__file__).with_name("index.html" if session.task["id"] == "risk_collect" else "board.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, value, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/state":
                try:
                    if isinstance(session, Replay):
                        query = parse_qs(url.query)
                        episode = int(query["episode"][0]) if "episode" in query else None
                        state = session.state(episode, int(query.get("step", [0])[0]))
                    else:
                        state = session.state()
                    self.respond(200, state)
                except (ValueError, KeyError) as exc:
                    self.respond(400, {"error": str(exc)})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if isinstance(session, Replay):
                self.respond(405, {"error": "replay is read-only"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 65536:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length) or b"{}")
                extra = {}
                if self.path == "/api/new":
                    seed = body.get("seed")
                    if isinstance(seed, str) and seed.isdecimal():
                        seed = int(seed)
                    session.new_game(seed=seed, task_id=body.get("task_id"))
                elif self.path == "/api/action":
                    session.step(body["action"])
                elif self.path == "/api/hint":
                    session.hint(body.get("policy_id"))
                elif self.path == "/api/bot-step":
                    suggestion = session.hint(body.get("policy_id"))
                    session.step(suggestion["action"])
                    extra["last_decision"] = suggestion
                elif self.path == "/api/save":
                    extra["save_id"] = session.save_game()
                elif self.path == "/api/load":
                    session.load_game(body["save_id"])
                elif self.path == "/api/challenge":
                    session.run_challenge(body.get("policy_id"))
                elif self.path == "/api/export":
                    extra["replay_path"] = session.export_replay()
                else:
                    self.respond(404, {"error": "not found"})
                    return
                self.respond(200, {**session.state(), **extra})
            except (InvalidAction, ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

    class Server(HTTPServer):
        def server_close(self):
            if isinstance(session, PlaySession) and session.registry:
                session.registry.close()
            super().server_close()
    return Server(("127.0.0.1", port), Handler)


def serve(*, config=None, trajectory=None, port=8765):
    server = make_server(config=config, trajectory=trajectory, port=port)
    print(f"BoardBench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
