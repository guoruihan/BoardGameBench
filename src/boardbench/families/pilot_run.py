"""Trusted pilot schedule and artifact lifecycle; execution is supplied by an adapter.

Adapters MUST enforce ``deadline_unix`` and budgets in their process supervisor,
use ledger_path across development resumes, and keep all trusted state outside
workspace/home. This module never invokes a model or GT during checkpoint checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


RESUME_PROMPT = "Continue the task using the existing workspace and remaining budget. Save reusable material in submission/ and checkpoint it. You may finish when ready."
DEVELOPMENT_PROMPT = """Develop and save reusable decision capability to maximize win rate in new Harmonies games under the supplied budgets. Read task_config.json, the rules and SDK documentation. You may play, inspect acquired history, analyze, write and run programs, build your own simulator and perform explicit parameter experiments. Activities share one development budget; no fixed order or reflection schedule is required. Access the real environment only through the SDK. Hidden state, GT implementation, opponent weights and other model services are unavailable. Do not train additional neural policy/value models or change the base model. Target deployment is L20/C120/GPU=0. Each test is independent and may use different stated budgets. Read the current observation before each real action. Save reusable notes and optional code in submission/ with manifest.json and START_HERE.md; checkpoint saves a format-validated copy. The test does not inherit your conversation. Use jobs.start for long experiments or packaging; managed tasks survive model exit within the original compute budget. Publish each runnable version with submission.publish before long reports. There is no extra model call for packaging when budgets expire. You may finish at any time."""
TEST_PROMPT = "Play this independent Harmonies match to maximize win probability using the provided SDK, rules and task_config.json. Read submission/START_HERE.md. You may use native tools and modify files within this match. Read the current observation before each action. Only this match's stated budgets are available; no prior conversation is available. Do not train additional neural policy/value models or access hidden state, GT implementation, opponent weights or other model services."


@dataclass(frozen=True)
class Budget:
    max_gt_games: int
    max_model_requests: int
    max_input_tokens: int
    max_output_tokens: int
    max_output_tokens_per_request: int
    active_wall_seconds: int
    cpu_cores: int
    cpu_core_seconds: int
    memory_gib: int
    max_input_tokens_per_request: int = 131072
    request_timeout_seconds: int = 120
    decision_timeout_seconds: int = 120
    gpu_enabled: bool = False
    cpu_enforcement: str = "supervised_polling_not_kernel_hard"


DEVELOPMENT_BUDGET = Budget(10, 100, 2_000_000, 256_000, 8192, 7200, 8, 57600, 32)
DEPLOYMENT_BUDGETS = {
    name: Budget(1, calls, inp, out, 4096, 1200, 2, 120, 8)
    for name, calls, inp, out in (("L10", 10, 128000, 8000), ("L20", 20, 256000, 16000), ("L40", 40, 512000, 32000))
}


@dataclass(frozen=True)
class PilotConfig:
    opponent_id: str
    opponent_sha256: str
    environment_revision: str
    codex_version: str
    deadline_unix: float
    seed_namespace: str = "harmonies-pilot-v2-20261115"
    model_id: str = "gpt-6-astra"
    reasoning_effort: str = "medium"
    protocol_version: str = "pilot_v2_20261115_c4"
    game_config: str = "side_a_base32_no_spirits_duo"
    test_concurrency: int = 4
    calibration_seeds: tuple[int, ...] = ()
    deployment_profiles: tuple[str, ...] = ("L10", "L20", "L40")

    def __post_init__(self):
        if (not isinstance(self.deployment_profiles, tuple) or not self.deployment_profiles
                or any(not isinstance(profile, str) or profile not in DEPLOYMENT_BUDGETS for profile in self.deployment_profiles)
                or len(set(self.deployment_profiles)) != len(self.deployment_profiles)):
            raise ValueError("deployment_profiles must be a nonempty unique tuple of L10/L20/L40")
        if not 1 <= self.test_concurrency <= 4:
            raise ValueError("test concurrency must be between one and four")
        if self.calibration_seeds and (len(self.calibration_seeds) != 5 or len(set(self.calibration_seeds)) != 5):
            raise ValueError("exactly five distinct calibration seeds required")
        for value in (self.opponent_id, self.environment_revision, self.codex_version):
            if not value or "REQUIRED" in value:
                raise ValueError("provenance must be resolved before preparing the pilot")
        if len(self.opponent_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.opponent_sha256):
            raise ValueError("opponent_sha256 must be a lowercase SHA256")
        if self.model_id not in {"gpt-6-astra", "gpt-5.6-sol"} or self.reasoning_effort != "medium":
            raise ValueError("this pilot requires a supported model with medium reasoning")


def seed_for(namespace: str, domain: str, index: int) -> int:
    """Disjoint integer ranges, including if a hash suffix collides across domains."""
    domains = {"acceptance": 1, "development": 2, "calibration": 3, "formal": 4}
    suffix = int.from_bytes(hashlib.sha256(f"{namespace}:{index}".encode()).digest()[:4], "big")
    return (domains[domain] << 40) | (suffix << 8) | index


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    profile: str
    seed: int
    seat: int
    control: bool


def default_queue(config: PilotConfig) -> list[MatchSpec]:
    return [MatchSpec(f"{arm}-{profile}-s{i}-p{seat}", profile,
                      (config.calibration_seeds[i] if config.calibration_seeds else seed_for(config.seed_namespace, "calibration", i)), seat, arm == "control")
            for arm, profiles in (("developed", config.deployment_profiles), ("control", ("L20",)))
            for profile in profiles for i in range(5) for seat in range(2)]


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("entry must be a relative file path")
    path = root / relative
    if ".." in Path(relative).parts or path.is_symlink() or not path.is_file():
        raise ValueError("entry is missing or unsafe")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("entry escapes submission")
    return path


def validate_submission(root: Path) -> dict:
    """Format-only validation, never execute participant code or consult GT."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("submission must be a real directory")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("links and special files are forbidden")
        if any(part in {".codex", ".git", ".ssh", ".aws", "__pycache__"} for part in path.relative_to(root).parts):
            raise ValueError("private state/cache directories cannot be submitted")
    manifest = json.loads(_safe_file(root, "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("entry_mode") not in {"codex", "program"}:
        raise ValueError("unsupported manifest schema or entry mode")
    if manifest.get("startup_file") != "START_HERE.md":
        raise ValueError("startup_file must be START_HERE.md")
    _safe_file(root, manifest["startup_file"])
    if manifest["entry_mode"] == "program":
        _safe_file(root, manifest.get("program_entry"))
    if manifest.get("fallback_entry") is not None:
        _safe_file(root, manifest["fallback_entry"])
    dependencies = manifest.get("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(dep, str) or not dep.strip() for dep in dependencies):
        raise ValueError("dependencies must be a list of nonempty installed dependency names")
    return manifest


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            content = path.read_bytes()
            name = path.relative_to(root).as_posix().encode()
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


class Checkpoints:
    def __init__(self, trusted_root: Path):
        import threading
        self.lock = threading.RLock()
        self.root = Path(trusted_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, submission: Path, installed_dependencies=None) -> str:
        with self.lock:
            return self._checkpoint(submission, installed_dependencies)

    def publish(self, submission: Path, entrypoint=None, installed_dependencies=None, admissible=lambda: True) -> str:
        with self.lock:
            return self._checkpoint(submission, installed_dependencies, entrypoint, admissible)

    def _checkpoint(self, submission: Path, installed_dependencies=None, entrypoint=None, admissible=lambda: True) -> str:
        if (self.root / "frozen.json").exists():
            raise RuntimeError("submission is already frozen")
        from .pilot_snapshot import safe_snapshot
        temporary = self.root / "pending"
        if temporary.exists():
            shutil.rmtree(temporary)
        safe_snapshot(submission, temporary)
        if entrypoint is not None:
            entry = _safe_file(temporary, entrypoint)
            if entry.suffix != '.py' and not os.access(entry, os.X_OK):
                raise ValueError('program_must_be_python_or_executable')
            manifest_path = temporary / 'manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            manifest.update(schema_version=1, entry_mode='program', program_entry=entrypoint,
                            startup_file='START_HERE.md')
            manifest.setdefault('dependencies', [])
            _write_json(manifest_path, manifest)
            if not (temporary/'START_HERE.md').exists():
                (temporary/'START_HERE.md').write_text('Run the program entry declared in manifest.json.\n')
        manifest = validate_submission(temporary)
        if installed_dependencies is not None and not set(manifest.get("dependencies", [])) <= set(installed_dependencies):
            shutil.rmtree(temporary)
            raise ValueError("dependency_not_installed")
        digest = tree_hash(temporary)
        target = self.root / digest
        if target.exists():
            shutil.rmtree(temporary)
        else:
            temporary.rename(target)
        latest = self.root / "latest.json"
        sequence = json.loads(latest.read_text()).get("sequence", 0) + 1 if latest.exists() else 1
        # Check the original phase budget immediately before committing the pointer.
        if not admissible():
            raise RuntimeError('submission_budget_exhausted')
        _write_json(latest, {"sha256": digest, "sequence": sequence,
                             "entry_mode": manifest['entry_mode'],
                             "program_entry": manifest.get('program_entry'),
                             "published_unix": time.time()})
        return digest

    def freeze(self) -> Path | None:
        with self.lock:
            return self._freeze()

    def _freeze(self) -> Path | None:
        frozen = self.root / "frozen.json"
        source = frozen if frozen.exists() else self.root / "latest.json"
        if not source.exists():
            return None
        record = json.loads(source.read_text())
        path = self.root / record["sha256"]
        validate_submission(path)
        if tree_hash(path) != record["sha256"]:
            raise ValueError("checkpoint content changed")
        _write_json(frozen, record)
        return path


def blank_submission(path: Path):
    path.mkdir(parents=True, exist_ok=False)
    _write_json(path / "manifest.json", {"schema_version": 1, "entry_mode": "codex", "startup_file": "START_HERE.md", "dependencies": []})
    (path / "START_HERE.md").write_text("Read the supplied task_config.json, rules and SDK documentation to access the current match.\n")


@dataclass(frozen=True)
class PhaseContext:
    phase_id: str
    workspace: Path
    home: Path
    ledger_path: Path
    budget: Budget
    deadline_unix: float
    prompt: str
    checkpoints: Checkpoints | None = None


@dataclass(frozen=True)
class DevelopmentResult:
    session_id: str | None
    finished: bool = False
    environment_progress: int = 0  # Monotone trusted GT event count, not participant report.
    stop_reason: str | None = None


class ExecutionAdapter(Protocol):
    def run_development(self, context: PhaseContext, resume_session: str | None) -> DevelopmentResult: ...
    def run_match(self, context: PhaseContext, match: MatchSpec) -> dict: ...


class PilotRunner:
    """Sequential durable queue. Interrupted/infrastructure-failed phases never retry.

    The adapter owns supervised execution and model/GT ledgers. Its development
    callback may invoke context.checkpoints.checkpoint from a trusted RPC handler.
    Test callbacks never receive a checkpoint handler. No test data enters development.
    """
    def __init__(self, root: Path, config: PilotConfig, adapter: ExecutionAdapter,
                 clock: Callable[[], float] = time.time):
        self.root, self.config, self.adapter, self.clock = Path(root), config, adapter, clock
        self.root.mkdir(parents=True, exist_ok=True)
        config_path = self.root / "config.json"
        serialized_config = json.loads(json.dumps(asdict(config)))
        if config_path.exists() and json.loads(config_path.read_text()) != serialized_config:
            raise ValueError("cannot change a persisted pilot configuration")
        _write_json(config_path, serialized_config)
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"development": "pending", "matches": {}, "status": "pending"}
        self.checkpoints = Checkpoints(self.root / "checkpoints")
        _write_json(self.root / "schedule.json", [asdict(m) for m in default_queue(config)])

    def _event(self, event: str, **fields):
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": self.clock(), "event": event, **fields}, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _write_json(self.state_path, self.state)

    def _context(self, phase: str, budget: Budget, submission: Path | None, development=False) -> PhaseContext:
        base = self.root / "phases" / phase
        base.mkdir(parents=True, exist_ok=False)
        work, home = base / "work", base / "home"
        work.mkdir()
        home.mkdir()
        if submission is None:
            blank_submission(work / "submission")
        else:
            shutil.copytree(submission, work / "submission")
        _write_json(work / "task_config.json", {
            "phase": "development" if development else "test", "budget": asdict(budget),
            "game_config": self.config.game_config, "information": "known",
            "model_id": self.config.model_id, "reasoning_effort": self.config.reasoning_effort,
            "target_deployment": "L20_C120_G0", "target_budget": asdict(DEPLOYMENT_BUDGETS["L20"]),
        })
        return PhaseContext(phase, work, home, base / "ledger.json", budget,
                            min(self.config.deadline_unix, self.clock() + budget.active_wall_seconds),
                            DEVELOPMENT_PROMPT if development else TEST_PROMPT,
                            self.checkpoints if development else None)

    def _run_tests(self, frozen: Path):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        pending = iter(m for m in default_queue(self.config) if m.match_id not in self.state['matches'])
        active = {}
        exhausted = False
        with ThreadPoolExecutor(max_workers=self.config.test_concurrency) as executor:
            while active or not exhausted:
                while len(active) < self.config.test_concurrency and not exhausted and self.state['status'] == 'running':
                    if self.clock() >= self.config.deadline_unix:
                        self.state['status'] = 'deadline'
                        self._event('deadline')
                        exhausted = True
                        break
                    match = next(pending, None)
                    if match is None:
                        exhausted = True
                        break
                    self.state['matches'][match.match_id] = 'running'
                    self._event('match_start', **asdict(match))
                    context = self._context(match.match_id, DEPLOYMENT_BUDGETS[match.profile], None if match.control else frozen)
                    initial_hash = tree_hash(context.workspace / 'submission')
                    active[executor.submit(self.adapter.run_match, context, match)] = (match, context, initial_hash)
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    match, context, initial_hash = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {'status': 'infrastructure_failure', 'error_type': type(error).__name__, 'error': str(error)}
                    _write_json(context.workspace.parent / 'result.json', result)
                    with (self.root / 'results.jsonl').open('a') as stream:
                        stream.write(json.dumps({**asdict(match), 'submission_sha256': initial_hash, 'result': result}) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    self.state['matches'][match.match_id] = 'finished'
                    self._event('match_end', match_id=match.match_id)
                    if result.get('status') == 'infrastructure_failure':
                        self.state['status'] = 'infrastructure_failure'
                        exhausted = True
                        self._event('queue_paused_no_retry', match_id=match.match_id)

    def run_frozen(self, frozen_submission: Path) -> dict:
        """Evaluate an existing trusted frozen checkpoint in a fresh cohort only.

        This imports bytes, never develops or selects a strategy. Existing run
        state, phase ledgers and checkpoint history cannot be reset by this API;
        ordinary ``run()`` remains the recovery/read-only resume entry point.
        """
        import fcntl
        with (self.root / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.state_path.exists() or (self.root / 'phases').exists()
                    or (self.root / 'results.jsonl').exists()
                    or (self.root / 'events.jsonl').exists()
                    or any(self.checkpoints.root.iterdir())):
                raise ValueError("frozen evaluation requires a fresh cohort; existing state cannot be reset")
            source = Path(os.path.abspath(frozen_submission))
            marker = source.parent / 'frozen.json'
            if marker.is_symlink() or not marker.is_file():
                raise ValueError("source must be an existing trusted frozen checkpoint")
            record = json.loads(marker.read_text())
            expected = record.get('sha256')
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in '0123456789abcdef' for c in expected)
                    or source.name != expected):
                raise ValueError("source checkpoint path does not match its frozen hash")
            source_config = source.parent.parent / 'config.json'
            if source_config.exists():
                previous = json.loads(source_config.read_text())
                if previous.get('deadline_unix') != self.config.deadline_unix:
                    raise ValueError("frozen cohort must preserve the original deadline")
            validate_submission(source)
            if tree_hash(source) != expected:
                raise ValueError("source frozen checkpoint content changed")
            try:
                copied_hash = self.checkpoints.checkpoint(source)
                if copied_hash != expected:
                    raise ValueError("source changed while importing frozen checkpoint")
                local = self.checkpoints.freeze()
            except BaseException:
                # The lock and fresh-cohort gate establish sole ownership of
                # this previously empty destination; source remains untouched.
                shutil.rmtree(self.checkpoints.root)
                self.checkpoints.root.mkdir()
                raise
            self.state['development'] = 'reused_frozen'
            self.state['frozen_submission'] = str(local)
            self.state['frozen_submission_sha256'] = expected
            self.state['source_frozen_submission'] = str(source)
            self._event('frozen_submission_reused', sha256=expected,
                        source_checkpoint=str(source), source_frozen_record=str(marker),
                        evaluation_only=True)
            return self._run()

    def run(self) -> dict:
        import fcntl
        with (self.root / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
            return self._run()

    def _run(self) -> dict:
        if self.state["status"] in {"complete", "infrastructure_failure", "deadline", "no_valid_submission"}:
            return self.state
        if self.state["development"] == "running" or "running" in self.state["matches"].values():
            self.state["status"] = "infrastructure_failure"
            self.state["frozen_submission"] = str(self.checkpoints.freeze() or "")
            self._event("interrupted_no_retry")
            return self.state
        if self.clock() >= self.config.deadline_unix:
            self.state["status"] = "deadline"
            self.state["frozen_submission"] = str(self.checkpoints.freeze() or "")
            self._event("deadline")
            return self.state
        self.state["status"] = "running"
        try:
            if self.state["development"] == "pending":
                self.state["development"] = "running"
                self._event("development_start")
                context = self._context("development", DEVELOPMENT_BUDGET, None, True)
                session, empty, gt_progress = None, 0, 0
                while self.clock() < context.deadline_unix:
                    before = _workspace_progress(context.workspace)
                    previous_checkpoint = (self.checkpoints.root / "latest.json").read_bytes() if (self.checkpoints.root / "latest.json").exists() else b""
                    result = self.adapter.run_development(context, session)
                    self._event("development_cycle", **asdict(result))
                    if result.finished or result.stop_reason:
                        break
                    if not result.session_id:
                        raise RuntimeError("unfinished development needs an explicit session ID")
                    after_checkpoint = (self.checkpoints.root / "latest.json").read_bytes() if (self.checkpoints.root / "latest.json").exists() else b""
                    progress = (before != _workspace_progress(context.workspace) or result.environment_progress > gt_progress or previous_checkpoint != after_checkpoint)
                    empty = 0 if progress else empty + (session is not None)
                    gt_progress, session = result.environment_progress, result.session_id
                    if empty >= 2:
                        self._event("early_exit", consecutive_empty_resumes=empty)
                        break
                    context = PhaseContext(**{**context.__dict__, "prompt": RESUME_PROMPT})
                if hasattr(self.adapter, "finish_development"):
                    self.adapter.finish_development(context)
                self.state["development"] = "finished"
                self._event("development_end")
            frozen = self.checkpoints.freeze()
            self.state["frozen_submission"] = str(frozen or "")
            if frozen is None:
                self.state["status"] = "no_valid_submission"
                self._event("no_valid_submission")
                return self.state
            self._event("submission_frozen", sha256=frozen.name)
            self._run_tests(frozen)
            if self.state["status"] != "running":
                return self.state
            self.state["status"] = "complete"
            self._event("complete")
        except Exception as error:
            self.state["status"] = "infrastructure_failure"
            self.state["frozen_submission"] = str(self.checkpoints.freeze() or "")
            self._event("infrastructure_failure", error_type=type(error).__name__, error=str(error))
            raise
        return self.state


# The control SDK is deliberately standalone and exposes no trusted paths.
_CONTROL_SDK = '''"""Submission and registered-program controls, with no strategy or GT access."""
import http.client, json, os, socket
class Connection(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX)
        self.sock.settimeout(120)
        self.sock.connect(os.environ['PILOT_CONTROL_SOCKET'])
def request(method, **fields):
    conn=Connection('localhost', timeout=120)
    try:
        conn.request('POST','/control',json.dumps(dict(method=method,**fields)),
                     {'Authorization':'Bearer '+os.environ['PILOT_CONTROL_TOKEN']})
        result=json.loads(conn.getresponse().read())
    finally: conn.close()
    if not result.get('ok'): raise RuntimeError(result.get('error','control_failed'))
    return result
def checkpoint(): return request('checkpoint')
def finish(): return request('finish')
def register_program(entry): return request('register_program',entry=entry)
'''


def prepare_public_material(destination: Path, sdk_client: Path, rules_files: list[Path]):
    """Copy only explicitly selected public material; never mount repository dirs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(sdk_client, destination / "pilot_client.py")
    (destination / "pilot_control.py").write_text(_CONTROL_SDK)
    (destination / 'jobs.py').write_text("from pilot_control import request\ndef start(argv): return request('jobs_start',argv=argv)['job_id']\ndef status(job_id=None): return request('jobs_status',job_id=job_id)\ndef output(job_id,stream='stdout',offset=0): return request('jobs_output',job_id=job_id,stream=stream,offset=offset)\n")
    (destination / 'submission.py').write_text("from pilot_control import request\ndef publish(directory='submission',entrypoint=None): return request('publish',directory=directory,entrypoint=entrypoint)\n")
    rules = destination / "rules"
    rules.mkdir()
    for source in rules_files:
        if not source.is_file() or source.is_symlink() or (rules / source.name).exists():
            raise ValueError("public rules must be explicit unique regular files")
        shutil.copyfile(source, rules / source.name)
    (destination / "SDK.md").write_text(
        "# Environment interface\n"
        "Runtime: Python 3.8 standard library and GCC/G++ are available; no NumPy/PyTorch or package downloads. "
        "Public rules/catalog remain fully readable under /bench/rules/. Snapshot limit: 64 MiB and 10,000 entries.\n"
        "Import `PilotClient` from `pilot_client` (PYTHONPATH is preconfigured). "
        "Development: call `new_game()` to start a game; tests start with one game already active. "
        "`status()`, `observe()`, `legal_actions(offset=0,limit=64)`, `history(offset=0,limit=64)`, "
        "and `budget()` read authorized information. `act(action_index,version)` uses the global "
        "index from the paginated actions and the current observation version. Read a fresh observation "
        "before each action. `run(policy,max_actions=None)` loops your callable over current observation "
        "and all current legal actions; return a global action index or NEED_AGENT. "
        "`abort()` consumes the current game; it does not refund its slot.\n\n"
        "During development, `import jobs; job_id=jobs.start(['python3','finalize.py'])` starts a managed "
        "task in /work in the same sandbox. `jobs.status(job_id)` and `jobs.output(job_id,stream='stdout',offset=0)` "
        "query it. Tasks share ALL original phase budgets and survive model-controller exit; no retries. "
        "After controller exit, new jobs are refused, but existing tasks may finish and publish. "
        "Logs retain at most 4 MiB per stream. These job APIs are development-only.\n"
        "`import submission; submission.publish(directory='candidate',entrypoint='play.py')` atomically "
        "saves a validated immutable snapshot AND its evaluation entry. Entry is Python or executable. "
        "Omit entrypoint to use the directory's existing manifest. Failure preserves the old submission. "
        "This does not run your program or test playing strength. Run minimal checks inside the sandbox first. "
        "Publish each runnable version you are willing to deliver BEFORE lengthy reports/validation. "
        "Later work cannot change published copies; only a successful publish replaces them. "
        "All jobs end or the original budget expires: freeze the latest successful submission.\n\n"
        "Import `pilot_control`. During development `checkpoint()` validates submission/ manifest "
        "and paths, then saves a snapshot without playing or scoring it. `finish()` ends development. "
        "`register_program('submission/path.py')` declares an optional existing program to execute "
        "if the model controller exits. A .py file runs with Python; another entry must be executable. "
        "Program execution uses the same game and remaining resource budgets.\n\n"
        "Submission manifest: schema_version=1, entry_mode='codex' or 'program', "
        "startup_file='START_HERE.md', dependencies=[]; program mode requires program_entry "
        "as a relative file path within submission/. Optional fallback_entry is also a relative "
        "file path. Only preinstalled declared dependencies may be used. Code is optional. "
        "No hidden files, credentials or session directories belong in the submission.\n"
    )


class NativePilotAdapter:
    """Native CLI + GT + supervisor integration, with an injected metered gateway.

    ``gateway_account(phase_id, budget)`` returns a persistent account token;
    repeated phase IDs MUST resolve to the same account without resetting usage.
    ``gateway_status(phase_id)`` returns trusted usage for logging and SDK budget.
    Gateway HTTP is exposed only through a private Unix-socket bridge. Real
    upstream endpoints and credentials are never placed in participant material.
    """
    def __init__(self, *, gt_service, gt_socket: Path, gateway_socket: Path,
                 sdk_dir: Path, gateway_account: Callable[[str, Budget], str],
                 gateway_status: Callable[[str], dict], supervisor=None,
                 installed_dependencies: tuple[str, ...] = (), test_concurrency: int = 4,
                 development_phase: str = "development", model_id: str = "gpt-6-astra", gateway_close=None):
        if model_id not in {"gpt-6-astra", "gpt-5.6-sol"}:
            raise ValueError("unsupported pilot model")
        self.model_id = model_id
        self.gateway_close = gateway_close
        from .pilot_supervisor import SandboxSupervisor, minimal_runtime
        self.gt, self.gt_socket = gt_service, Path(gt_socket)
        self.gateway_socket, self.sdk_dir = Path(gateway_socket), Path(sdk_dir)
        self.gateway_account, self.gateway_status = gateway_account, gateway_status
        self.supervisor = supervisor or SandboxSupervisor()
        self.runtime = minimal_runtime(include_codex=True, include_compiler=True)
        self.installed_dependencies = set(installed_dependencies)
        if development_phase not in ("development", "smoke"):
            raise ValueError("unsupported development seed domain")
        self.development_phase = development_phase
        self.sessions: dict[str, dict] = {}
        import queue
        from .pilot_supervisor import physical_cpu_ids
        cores = physical_cpu_ids(test_concurrency * 2)
        self.core_slots = queue.Queue()
        for index in range(test_concurrency):
            self.core_slots.put(cores[2 * index:2 * index + 2])
        self.phase_cores = {}

    def _open_phase(self, context: PhaseContext, match: MatchSpec | None):
        from .pilot_gt import create_run
        if context.phase_id in self.sessions:
            return self.sessions[context.phase_id]
        record_path = context.workspace.parent / "services.json"
        if record_path.exists():
            raise RuntimeError("service phase already exists; interrupted phases cannot auto-retry")
        credentials = create_run(self.gt.root, phase=self.development_phase if match is None else "calibration",
                                 pair_index=0 if match is None else int(match.match_id.split('-s')[1].split('-')[0]),
                                 seat=0 if match is None else match.seat, budget=asdict(context.budget))
        record = {"gt": credentials, "gateway_token": self.gateway_account(context.phase_id, context.budget),
                  "finished": False, "registered_program": None}
        _write_json(record_path, record)
        record_path.chmod(0o600)
        self.sessions[context.phase_id] = record
        codex = context.workspace / ".codex"
        codex.mkdir()
        (codex / "config.toml").write_text('model = "' + self.model_id + '"\n' + '''
model_reasoning_effort = "medium"
model_provider = "pilot_gateway"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
model_auto_compact_token_limit = 32000
[model_providers.pilot_gateway]
name = "Metered benchmark gateway"
base_url = "http://127.0.0.1:18080/v1"
env_key = "PILOT_MODEL_TOKEN"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
''')
        (context.workspace / "TASK.md").write_text(context.prompt + "\n\nRead /bench/SDK.md and /bench/rules/.\n")
        if match is not None:
            started = self.gt.request(credentials['token'], 'trusted-initial-game',
                                      {"method": "new_game", "generation": credentials['generation']})
            if not started.get('ok'):
                raise RuntimeError(f"test initialization failed: {started.get('error')}")
        return record

    def _control_server(self, context, record):
        import http.server
        import secrets
        import socketserver
        import threading
        adapter = self
        token = secrets.token_urlsafe(32)
        # Short socket path avoids the platform's 108-byte sockaddr_un limit.
        import tempfile
        socket_directory = Path(tempfile.mkdtemp(prefix="pilot-control-"))
        socket_path = socket_directory / "control.sock"

        class Handler(http.server.BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def log_message(self, *args):
                pass

            def do_POST(self):
                result = {"ok": False, "error": "invalid_request"}
                if self.path != '/control' or self.headers.get('Authorization') != 'Bearer ' + token:
                    self.send_error(403)
                    return
                control_started = time.thread_time()
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 65536:
                        raise ValueError('invalid_request')
                    body = json.loads(self.rfile.read(size))
                    method = body.get('method')
                    manager = record.get('jobs')
                    if isinstance(method, str) and method.startswith('broker_'):
                        import socket, struct
                        pid = struct.unpack('3i', self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[0]
                        if manager is None or not manager.broker_allowed(pid):
                            raise ValueError('broker_identity_denied')
                        if method == 'broker_poll':
                            result = dict(ok=True, jobs=manager.poll(), stop_model=record['finished'])
                        elif method == 'broker_pending':
                            result = dict(ok=True, active=manager.active())
                        elif method == 'broker_event':
                            manager.event(**{k:v for k,v in body.items() if k != 'method'})
                            if body.get('kind') == 'controller_exit' and manager.jobs and adapter.gateway_close:
                                adapter.gateway_close(context.phase_id)
                            result = dict(ok=True)
                        else:
                            raise ValueError('unsupported_broker_method')
                    elif method in ('jobs_start', 'jobs_status', 'jobs_output') and manager is not None:
                        if method == 'jobs_start':
                            result = dict(ok=True, **manager.start(body.get('argv')))
                        elif method == 'jobs_status':
                            result = dict(ok=True, **manager.status(body.get('job_id')))
                        else:
                            result = dict(ok=True, **manager.output(body.get('job_id'), body.get('stream','stdout'), body.get('offset',0)))
                    elif method == 'publish' and context.checkpoints is not None:
                        directory = body.get('directory', 'submission')
                        if not isinstance(directory, str) or Path(directory).is_absolute() or '..' in Path(directory).parts:
                            raise ValueError('invalid_submission_directory')
                        digest = context.checkpoints.publish(context.workspace / directory,
                            body.get('entrypoint'), adapter.installed_dependencies,
                            lambda: record['publish_admissible'](time.thread_time()-control_started))
                        result = dict(ok=True, sha256=digest)
                    elif method == 'checkpoint' and context.checkpoints is not None:
                        digest = context.checkpoints.publish(context.workspace / 'submission',
                            installed_dependencies=adapter.installed_dependencies,
                            admissible=lambda: record.get('publish_admissible',lambda *_: True)(time.thread_time()-control_started))
                        result = {"ok": True, "sha256": digest}
                    elif method == 'finish' and context.checkpoints is not None:
                        record['finished'] = True
                        result = {"ok": True}
                    elif method == 'register_program':
                        entry = _safe_file(context.workspace, body.get('entry'))
                        if not entry.resolve().is_relative_to((context.workspace / 'submission').resolve()):
                            raise ValueError('program_must_be_in_submission')
                        if entry.suffix != '.py' and not os.access(entry, os.X_OK):
                            raise ValueError('program_must_be_python_or_executable')
                        record['registered_program'] = str(entry.relative_to(context.workspace))
                        result = {"ok": True}
                    else:
                        raise ValueError('unsupported_method')
                except (ValueError, KeyError, OSError, RuntimeError) as error:
                    result = {"ok": False, "error": str(error)}
                record['control_cpu_seconds'] = record.get('control_cpu_seconds', 0.) + time.thread_time() - control_started
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Server(socketserver.UnixStreamServer):
            pass

        server = Server(str(socket_path), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, socket_path, token

    def _execute(self, context, record, *, resume_session=None, program=None, match=False):
        from .pilot_supervisor import SandboxSpec, Limits, WRAPPER, physical_cpu_ids
        control_cpu_before = record.get('control_cpu_seconds', 0.)
        from .pilot_jobs import ManagedJobs
        compute = json.loads(context.ledger_path.read_text()) if context.ledger_path.exists() else {}
        live = {'cpu': compute.get('cpu_seconds',0.), 'wall': compute.get('active_wall_seconds',0.), 'rss':0}
        execution_started = time.monotonic()
        execution_active = True
        def admissible(inflight_cpu=0.):
            return (execution_active and time.time() < context.deadline_unix
                    and compute.get('active_wall_seconds',0.) + time.monotonic()-execution_started < context.budget.active_wall_seconds
                    and live['cpu'] + record.get('control_cpu_seconds',0.)-control_cpu_before + inflight_cpu < context.budget.cpu_core_seconds
                    and live['rss'] < context.budget.memory_gib*1024**3)
        record['publish_admissible'] = admissible
        manager = None
        if context.checkpoints is not None:
            number = record.get('job_queue_number',0)
            record['job_queue_number'] = number + 1
            manager = ManagedJobs(context.workspace.parent/f'jobs-{number}', admissible)
        record['jobs'] = manager
        control, control_path, control_token = self._control_server(context, record)
        credentials = record['gt']
        if program:
            credentials = self.gt.rotate_controller(credentials['run_id'], source='program')
            record['gt'] = credentials
            entry = _safe_file(context.workspace, program)
            argv = ['/usr/bin/python3', '/work/' + program] if entry.suffix == '.py' else ['/work/' + program]
        else:
            argv = [str(WRAPPER), 'exec']
            if resume_session:
                argv += ['resume']
            argv += ['--json', '--skip-git-repo-check', '--dangerously-bypass-approvals-and-sandbox']
            if resume_session:
                argv += [resume_session]
            argv += ['-']
        previous = record.setdefault('decision_clock', {'version': None, 'decision_start': time.time(), 'last_poll': 0.0, 'budget_poll': 0.0})

        def stop_requested():
            now = time.time()
            if now >= context.deadline_unix:
                return 'absolute_deadline'
            if record.get('control_cpu_seconds', 0.) - control_cpu_before >= context.budget.cpu_core_seconds:
                return 'cpu_limit'
            if record['finished'] and (manager is None or not manager.active()):
                return 'development_finished'
            if now - previous['last_poll'] < .2:
                return None
            previous['last_poll'] = now
            report = self.gt.report(credentials['run_id'])
            if now - previous['budget_poll'] >= 1:
                previous['budget_poll'] = now
                self.gt.update_budget(credentials['run_id'], {'limits': asdict(context.budget), 'model': self.gateway_status(context.phase_id), 'absolute_wall_remaining': max(0, context.deadline_unix - now)})
            if match and report['status'] != 'active':
                return 'game_finished'
            if report['version'] != previous['version']:
                previous.update(version=report['version'], decision_start=now)
            if match and now - previous['decision_start'] >= context.budget.decision_timeout_seconds:
                return 'decision_timeout'
            return None

        try:
            self.gt.update_budget(credentials['run_id'], {
                'limits': asdict(context.budget), 'model': self.gateway_status(context.phase_id),
                'compute': json.loads(context.ledger_path.read_text()) if context.ledger_path.exists() else {},
            })
            result = self.supervisor.run(SandboxSpec(
                argv=argv, work_dir=context.workspace, sdk_dir=self.sdk_dir,
                ledger_path=context.ledger_path, cpu_ids=self.phase_cores.get(context.phase_id) or physical_cpu_ids(context.budget.cpu_cores),
                limits=Limits(context.budget.cpu_core_seconds, context.budget.active_wall_seconds,
                              context.budget.memory_gib * 1024**3),
                runtime_mounts=self.runtime,
                sockets={'gt.sock': self.gt_socket, 'gateway.sock': self.gateway_socket, 'control.sock': control_path},
                tcp_bridges={18080: 'gateway.sock'},
                env={'PYTHONPATH': '/bench', 'PILOT_GT_SOCKET': '/run/bench/gt.sock',
                     'PILOT_GT_TOKEN': credentials['token'], 'PILOT_GT_GENERATION': str(credentials['generation']),
                     'PILOT_MODEL_TOKEN': record['gateway_token'], 'PILOT_CONTROL_SOCKET': '/run/bench/control.sock',
                     'PILOT_CONTROL_TOKEN': control_token},
                stdin=b'' if program else (context.prompt + '\nRead /bench/SDK.md and /bench/rules/.').encode(),
                stop_requested=stop_requested,
                managed_jobs=manager is not None,
                on_started=(lambda pid: setattr(manager,'sandbox_pid',pid)) if manager else None,
                on_sample=lambda cpu,wall,rss: live.update(cpu=cpu,wall=wall,rss=rss),
                extra_cpu_seconds=lambda: record.get('control_cpu_seconds',0.)-control_cpu_before,
            ))
        finally:
            execution_active = False
            if manager is not None:
                manager.close("sandbox_ended")
            control.shutdown()
            control.server_close()
            shutil.rmtree(control_path.parent)
        control_cpu = record.get('control_cpu_seconds', 0.) - control_cpu_before
        if control_cpu and context.ledger_path.exists():
            ledger = json.loads(context.ledger_path.read_text())
            ledger['cpu_seconds'] = ledger.get('cpu_seconds', 0.) + control_cpu
            ledger['control_cpu_seconds'] = ledger.get('control_cpu_seconds', 0.) + control_cpu
            _write_json(context.ledger_path, ledger)
        settlement_reason = self._await_gateway_settlement(context, result.reason)
        if settlement_reason is not None:
            result.reason = settlement_reason
        return result

    def _await_gateway_settlement(self, context: PhaseContext, controller_reason: str) -> str | None:
        """Wait after the supervised process exits, without resending generation.

        Pending reservations belong to the old controller even after its socket
        disconnects. Only this post-process drain wait is added to the cumulative
        wall ledger; supervised live-process time has already been charged.
        """
        gateway = self.gateway_status(context.phase_id)

        def pending(status):
            reservations = status.get('pending_reservations', {})
            calls = reservations.get('calls', 0) if isinstance(reservations, dict) else reservations
            return max(calls or 0, int(bool(status.get('preflighting'))))

        def unknown(status):
            return bool(status.get('frozen') or status.get('unknown_requests'))

        if not pending(gateway) and not unknown(gateway):
            return None
        ledger = json.loads(context.ledger_path.read_text()) if context.ledger_path.exists() else {}
        started = time.monotonic()
        previous = started
        event = {'controller_reason': controller_reason, 'started_unix': time.time(),
                 'initial_pending_calls': pending(gateway), 'wall_seconds': 0., 'status': 'waiting'}
        ledger.setdefault('gateway_settlement_waits', []).append(event)
        status_error = None
        while True:
            now = time.monotonic()
            elapsed = max(0., now - previous)
            previous = now
            ledger['active_wall_seconds'] = ledger.get('active_wall_seconds', 0.) + elapsed
            ledger['gateway_settlement_wall_seconds'] = ledger.get('gateway_settlement_wall_seconds', 0.) + elapsed
            event['wall_seconds'] += elapsed
            event['remaining_pending_calls'] = pending(gateway)
            remaining = min(context.budget.active_wall_seconds - ledger['active_wall_seconds'],
                            context.deadline_unix - time.time())
            reason = None
            if status_error or unknown(gateway):
                event['status'] = 'infrastructure_failure'
                reason = 'infrastructure_failure'
                if status_error:
                    event['status_error_type'] = status_error
            elif remaining <= 0:
                # No free wait past the phase's absolute or cumulative deadline,
                # even if the last polling interval also observed settlement.
                event['status'] = 'wall_limit'
                reason = 'wall_limit'
            elif not pending(gateway):
                event['status'] = 'settled'
            _write_json(context.ledger_path, ledger)
            if event['status'] != 'waiting':
                return reason
            time.sleep(min(.05, remaining))
            try:
                gateway = self.gateway_status(context.phase_id)
            except Exception as error:
                # Retain charged wait and fail closed; never retry generation or
                # run another controller when settlement cannot be observed.
                status_error = type(error).__name__

    @staticmethod
    def _session_id(stdout_path: str) -> str | None:
        for line in Path(stdout_path).read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get('type') == 'thread.started':
                return item.get('thread_id')
        return None

    def run_development(self, context: PhaseContext, resume_session: str | None) -> DevelopmentResult:
        record = self._open_phase(context, None)
        rid = record['gt']['run_id']
        session_id = resume_session
        model_limits = {'calls', 'input_tokens', 'output_tokens', 'max_input_tokens_estimate'}

        def local_limit():
            ledger = json.loads(context.ledger_path.read_text()) if context.ledger_path.exists() else {}
            if ledger.get('cpu_seconds', 0.) >= context.budget.cpu_core_seconds:
                return 'cpu_limit'
            if ledger.get('active_wall_seconds', 0.) >= context.budget.active_wall_seconds or time.time() >= context.deadline_unix:
                return 'wall_limit'
            return None

        def exhausted(gateway):
            remaining = gateway.get('remaining', {})
            return any(key in remaining and remaining[key] <= 0
                       for key in ('calls', 'input_tokens', 'output_tokens'))

        def finish(stop):
            if stop:
                self.gt.stop(rid, reason='development_truncated')
            report = self.gt.report(rid)
            return DevelopmentResult(session_id, finished=record['finished'],
                                     environment_progress=report['games'] * 10000 + report['steps'],
                                     stop_reason=stop)

        gateway = self.gateway_status(context.phase_id)
        if gateway.get('frozen'):
            self.gt.stop(rid, reason='infrastructure_failure')
            raise RuntimeError('development gateway is frozen; no automatic retry')
        stop = local_limit()
        if stop or record['finished']:
            return finish(stop or 'development_finished')
        blocked = exhausted(gateway) or record.get('development_model_blocked', False)
        if not blocked:
            before_events = len(gateway.get('events', []))
            result = self._execute(context, record, resume_session=resume_session)
            session_id = self._session_id(result.stdout_path) or session_id
            gateway = self.gateway_status(context.phase_id)
            if result.reason == 'infrastructure_failure' or gateway.get('frozen') or gateway.get('unknown_requests'):
                self.gt.stop(rid, reason='infrastructure_failure')
                raise RuntimeError('development gateway settlement failed; inspect controller and gateway logs')
            stop = local_limit() or (result.reason if result.reason in
                                    {'cpu_limit', 'wall_limit', 'absolute_deadline', 'rss_limit', 'development_finished'} else None)
            if stop or record['finished']:
                return finish(stop or 'development_finished')
            # Cumulative counters can hit their limit without a rejected next
            # request. Also detect a new refusal with positive residual tokens,
            # without confusing old, recovered limit events with current usage.
            refused = any(event.get('kind') == 'rejected' and event.get('limit') in model_limits
                          for event in gateway.get('events', [])[before_events:])
            blocked = exhausted(gateway) or refused
            if record.get('jobs') is not None and record['jobs'].jobs:
                return finish('model_limit_reached' if blocked else 'managed_jobs_completed')
            if not blocked:
                if result.reason == 'execution_error':
                    self.gt.stop(rid, reason='infrastructure_failure')
                    raise RuntimeError('native development execution failed; inspect controller and gateway logs')
                return finish(None if result.reason == 'exited' else result.reason)
        record['development_model_blocked'] = True
        program = record.get('registered_program')
        if not program:
            return finish('model_limit_reached')
        # Exactly one registered-program handoff, never another Codex request.
        # _execute rotates the existing GT controller lease to program source;
        # the context preserves the same workspace, ledger and absolute deadline.
        if record.get('development_program_started'):
            return finish('model_limit_reached')
        record['development_program_started'] = True
        result = self._execute(context, record, program=program)
        gateway = self.gateway_status(context.phase_id)
        if result.reason == 'infrastructure_failure' or gateway.get('frozen') or gateway.get('unknown_requests'):
            self.gt.stop(rid, reason='infrastructure_failure')
            raise RuntimeError('development gateway settlement failed during registered program execution')
        stop = local_limit() or (result.reason if result.reason != 'exited' else 'model_limit_reached')
        if result.reason == 'execution_error' and stop not in {'cpu_limit', 'wall_limit'}:
            stop = 'program_execution_error'
        # A program may explicitly checkpoint/finish using normal controls; the
        # runner never packages or creates a checkpoint on the participant's behalf.
        return finish('development_finished' if record['finished'] else stop)

    def finish_development(self, context: PhaseContext):
        record = self.sessions.get(context.phase_id)
        if record is not None:
            self.gt.stop(record['gt']['run_id'], reason='development_truncated')

    def run_match(self, context: PhaseContext, match: MatchSpec) -> dict:
        cores = self.core_slots.get()
        self.phase_cores[context.phase_id] = cores
        try:
            return self._run_match(context, match)
        finally:
            self.phase_cores.pop(context.phase_id, None)
            self.core_slots.put(cores)

    def _run_match(self, context: PhaseContext, match: MatchSpec) -> dict:
        from .pilot_gt import private_seed
        pair_index = int(match.match_id.split('-s')[1].split('-')[0])
        actual_seed = private_seed(self.gt.config['seed_key'], 'calibration', pair_index)
        if actual_seed != match.seed:
            raise ValueError('PilotConfig.calibration_seeds must match the trusted GT schedule')
        record = self._open_phase(context, match)
        manifest = validate_submission(context.workspace / 'submission')
        program = 'submission/' + manifest['program_entry'] if manifest['entry_mode'] == 'program' else None
        fallback = 'submission/' + manifest['fallback_entry'] if manifest.get('fallback_entry') else None
        rid = record['gt']['run_id']
        session_id = None
        empty_resumes = 0
        result = None
        attempted_programs = set()
        last_reason = 'no_controller'
        resume_prompt = "Continue this same match using the existing workspace, current observation and remaining budget. The match is not finished."

        def compute():
            return json.loads(context.ledger_path.read_text()) if context.ledger_path.exists() else {}

        def local_limit():
            ledger = compute()
            if ledger.get('cpu_seconds', 0.) >= context.budget.cpu_core_seconds:
                return 'cpu_exhausted'
            if ledger.get('active_wall_seconds', 0.) >= context.budget.active_wall_seconds or time.time() >= context.deadline_unix:
                return 'wall_exhausted'
            return None

        def finalize(status=None):
            if status is not None:
                self.gt.stop(rid, reason=status)
            report = self.gt.report(rid)
            response = {'status': status or report['status'], 'gt': report,
                        'model': self.gateway_status(context.phase_id), 'compute': compute(),
                        'last_controller': asdict(result) if result is not None else None,
                        'cpu_enforcement': 'supervised_polling_not_kernel_hard'}
            if status in {'cpu_exhausted', 'wall_exhausted', 'memory_exhausted', 'agent_failure'}:
                # Preserve raw GT outcome if an action finished just before final
                # wait4 revealed overrun; the classified result still counts as a loss.
                response.update(win_points=0., natural=False)
            return response

        def model_available(gateway):
            remaining = gateway.get('remaining', {})
            return not gateway.get('frozen') and all(remaining.get(key, 0) > 0 for key in ('calls', 'input_tokens', 'output_tokens'))

        while self.gt.report(rid)['status'] == 'active':
            limit = local_limit()
            if limit:
                return finalize(limit)
            gateway = self.gateway_status(context.phase_id)
            if gateway.get('frozen') or gateway.get('unknown_requests'):
                return finalize('infrastructure_failure')
            if program is None and not model_available(gateway):
                registered = record.get('registered_program') or fallback
                if registered and registered not in attempted_programs:
                    program = registered
                else:
                    break
            before_files = _workspace_progress(context.workspace)
            before_version = self.gt.report(rid)['version']
            resuming = program is None and session_id is not None
            execution_context = context
            if resuming:
                execution_context = PhaseContext(**{**context.__dict__, 'prompt': resume_prompt})
            if program:
                attempted_programs.add(program)
            result = self._execute(execution_context, record, program=program,
                                   resume_session=session_id if program is None else None, match=True)
            last_reason = result.reason
            if result.reason == 'infrastructure_failure':
                return finalize('infrastructure_failure')
            if program is None:
                session_id = self._session_id(result.stdout_path) or session_id
            # Check settled cumulative usage after EVERY controller, including
            # fallback programs and normally exited processes. Final wait4 and
            # checkpoint CPU can cross limits without supervisor reason changing.
            limit = local_limit()
            if limit:
                return finalize(limit)
            reason_limit = {'cpu_limit': 'cpu_exhausted', 'wall_limit': 'wall_exhausted',
                            'absolute_deadline': 'wall_exhausted', 'rss_limit': 'memory_exhausted'}.get(result.reason)
            if reason_limit:
                return finalize(reason_limit)
            gateway = self.gateway_status(context.phase_id)
            if gateway.get('frozen') or gateway.get('unknown_requests'):
                return finalize('infrastructure_failure')
            report = self.gt.report(rid)
            if report['status'] != 'active':
                return finalize()
            if result.reason == 'execution_error':
                # A model-quota refusal can end Codex cleanly from the protocol's
                # perspective; a crashing participant program is still a failure.
                if program is not None or not gateway.get('first_binding_limit'):
                    return finalize('agent_failure')
                registered = record.get('registered_program') or fallback
                if registered and registered not in attempted_programs:
                    program = registered
                    continue
                break
            if result.reason == 'decision_timeout':
                self.gt.fallback(rid, reason='decision_timeout', finish=False)
                if self.gt.report(rid)['status'] != 'active':
                    return finalize()
                record['gt'] = self.gt.rotate_controller(rid, source='program' if program else 'agent')
                if program is not None or session_id:
                    continue
                break
            if program is not None:
                # NEED_AGENT is a voluntary handoff, not an invalid move or
                # failed process. Other normal program exits use default takeover.
                needs_agent = any(line.strip() == 'NEED_AGENT' for line in Path(result.stdout_path).read_text().splitlines())
                if needs_agent and model_available(gateway):
                    program = None
                    record['gt'] = self.gt.rotate_controller(rid, source='agent')
                    continue
                break
            progress = (before_version != report['version'] or before_files != _workspace_progress(context.workspace))
            if resuming:
                empty_resumes = 0 if progress else empty_resumes + 1
            if empty_resumes >= 2 or not session_id:
                last_reason = 'early_exit' if empty_resumes >= 2 else 'missing_session'
                registered = record.get('registered_program') or fallback
                if registered and registered not in attempted_programs:
                    program = registered
                    continue
                break
            # Normal CLI final output with an active game resumes this exact
            # session; it is not a fresh match, quota reset or infrastructure retry.

        # Default moves run in trusted GT, never generate extra model calls and
        # cannot finish a game for free after cumulative CPU/wall exhaustion.
        while self.gt.report(rid)['status'] == 'active':
            limit = local_limit()
            if limit:
                return finalize(limit)
            self.gt.fallback(rid, reason=last_reason, finish=False)
        return finalize()


def _workspace_progress(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part.startswith('.') for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink():
            digest.update(relative.as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def export_known_catalog(destination: Path):
    """Trusted static export only; this is not a replacement for complete rules."""
    from .direct_play import public_catalog
    from .games.harmonies import Harmonies
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / 'catalog.json', public_catalog('harmonies'))
    _write_json(destination / 'identity.json', {key: value for key, value in Harmonies.describe().items() if key in {'game_id', 'rules_version', 'config_id', 'players'}})


def gateway_callbacks(gateway):
    """Translate protocol budget names to the gateway account contract."""
    def account(phase_id, budget):
        return gateway.create_account(phase_id, {
            'calls': budget.max_model_requests,
            'input_tokens': budget.max_input_tokens,
            'output_tokens': budget.max_output_tokens,
            'max_input_tokens': budget.max_input_tokens_per_request,
            'max_output_tokens': budget.max_output_tokens_per_request,
        })['token']
    return account, gateway.status
