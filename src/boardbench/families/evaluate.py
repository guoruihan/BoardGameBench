"""Frozen, resumable same-seed paired evaluation; confirmation never selects bots.

Run ``python -m boardbench.families.evaluate --help``. Source/model snapshots,
all attempts, and unstarted jobs are preserved in the evaluation directory.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import itertools
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import tarfile
import time
import uuid

from .base import FEATURE_DIM
from .play import write_json
from .registry import game_class

FORMAT = 'boardbench_frozen_evaluation_v1'
RETRYABLE = {'not_started', 'running', 'interrupted', 'deadline'}
KINDS = {'random', 'heuristic', 'linear', 'neural', 'harmonies_rl'}
STOP = False


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_reference(value, directory):
    # A portable policy package resolves every relative path against the JSON
    # that defines it, never against the caller's working directory.
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(directory)/path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'Expected a regular policy/model file: {value!r}')
    return path


def freeze_model(source, output, expected_sha=None):
    directory = Path(output)/'models'
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory/f'.{uuid.uuid4().hex}.part'
    digest = hashlib.sha256()
    try:
        with Path(source).open('rb') as original, temporary.open('wb') as copy:
            for chunk in iter(lambda: original.read(1024*1024), b''):
                digest.update(chunk)
                copy.write(chunk)
        sha = digest.hexdigest()
        if expected_sha and expected_sha != sha:
            raise ValueError('Source model checksum mismatch')
        target = directory/f'{sha}.pt'
        if target.exists():
            if file_hash(target) != sha:
                raise ValueError('Existing frozen model has changed')
            temporary.unlink()
        else:
            temporary.replace(target)
            target.chmod(0o444)
        return dict(relative_path=str(target.relative_to(output)), sha256=sha,
                    bytes=target.stat().st_size, original_path=str(source))
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_bots(bot_file, output, game):
    """Resolve JSON references once and copy actual model bytes, not symlinks."""
    bot_file, output = Path(bot_file).resolve(), Path(output).resolve()
    supplied = read_json(bot_file)
    if not isinstance(supplied, dict) or len(supplied) < 2:
        raise ValueError('Bots JSON must map at least two distinct names to specs')
    if any(not isinstance(name, str) or not name.strip() for name in supplied):
        raise ValueError('Bot names must be nonempty strings')
    cls = game_class(game)
    bots = {}
    for name in sorted(supplied):
        spec = deepcopy(supplied[name])
        directory = bot_file.parent
        references = []
        visited = set()
        while isinstance(spec, str) or (isinstance(spec, dict) and 'spec_file' in spec):
            if isinstance(spec, dict) and set(spec) != {'spec_file'}:
                raise ValueError('spec_file references cannot silently override artifact fields')
            source = resolve_reference(spec if isinstance(spec, str) else spec['spec_file'], directory)
            if source in visited:
                raise ValueError('Cyclic policy specification reference')
            visited.add(source)
            content = source.read_bytes()
            references.append(dict(path=str(source), sha256=hashlib.sha256(content).hexdigest()))
            spec = json.loads(content)
            directory = source.parent
        if not isinstance(spec, dict):
            raise ValueError(f'{name}: expected policy object')
        spec.setdefault('kind', 'heuristic')
        if spec['kind'] not in KINDS:
            raise ValueError(f'{name}: unsupported policy kind')
        for key, expected in (('game', cls.game_id), ('config', cls.config_id),
                              ('rules_version', cls.rules_version)):
            if key in spec and spec[key] != expected:
                raise ValueError(f'{name}: policy game/config/rules mismatch')
            spec[key] = expected
        model = None
        if spec['kind'] in ('neural', 'harmonies_rl'):
            if spec['kind'] == 'harmonies_rl' and game != 'harmonies':
                raise ValueError('harmonies_rl is only valid for Harmonies')
            source = resolve_reference(spec['path'], directory)
            model = freeze_model(source, output, spec.get('sha256'))
            spec['path'] = str(output/model['relative_path'])
            spec['sha256'] = model['sha256']
        else:
            if 'path' in spec:
                raise ValueError('Feature policies use inline weights; use spec_file for a JSON artifact')
            if spec['kind'] == 'linear' and 'weights' not in spec:
                raise ValueError('A linear policy requires its actual frozen weights')
            spec.setdefault('weights', list(cls.default_weights))
            if len(spec['weights']) != FEATURE_DIM or not all(
                    isinstance(w, (int, float)) and math.isfinite(w) for w in spec['weights']):
                raise ValueError('Expected 24 finite policy weights')
        identity = deepcopy(spec)
        if model:
            identity['path'] = 'sha256:'+model['sha256']
        bots[name] = dict(spec=spec, identity_sha256=canonical_hash(identity),
                          model=model, spec_sources=references)
    return bots


def source_files():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    suffixes = {'.py', '.json', '.txt', '.cpp', '.h', '.c', '.toml'}
    files = [p for p in package.rglob('*') if p.is_file() and p.suffix in suffixes
             and '__pycache__' not in p.parts]
    if (root/'pyproject.toml').exists():
        files.append(root/'pyproject.toml')
    return root, {str(p.relative_to(root)): p for p in sorted(files)}


def code_identity():
    root, paths = source_files()
    hashes = {name: file_hash(path) for name, path in paths.items()}
    try:
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                                            text=True, stderr=subprocess.DEVNULL).strip()
        # An immutable git archive can live inside a changing checkout's outputs/.
        # The enclosing checkout's HEAD is not the archive's source revision.
        if Path(git_root).resolve() != root.resolve():
            commit, dirty = None, None
        else:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                 cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    dependencies = {}
    for name in ('numpy', 'torch'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return dict(git_commit=commit, tracked_dirty=dirty, files=hashes,
                content_sha256=canonical_hash(hashes), python=platform.python_version(),
                dependencies=dependencies, platform=platform.platform())


def archive_source(output):
    identity = code_identity()
    _, paths = source_files()
    target = Path(output)/'source.tar.gz'
    temporary = target.with_suffix('.part')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in paths.items():
            archive.add(path, arcname=name, recursive=False)
    temporary.replace(target)
    identity['archive_sha256'] = file_hash(target)
    return identity


def schedule_matches(game, bot_names, seeds):
    names = sorted(bot_names)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError('At least two uniquely named bots required')
    if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Environment seeds must be nonempty unique integers')
    rows = []
    # Round-robin seeds first: a deadline gives all pairs an early chance to run.
    for seed in seeds:
        for pair in itertools.combinations(names, 2):
            for orientation in (0, 1):
                seats = list(pair if orientation == 0 else pair[::-1])
                identity = dict(game=game, pair=list(pair), seed=seed, orientation=orientation,
                                first_player=0, seat_names=seats)
                rows.append(dict(identity, match_id=canonical_hash(identity)[:24]))
    return rows


def prepare_evaluation(game, bot_file, output, seeds, phase, max_seconds=120.,
                       max_decisions=10000, record=False, validate=False,
                       exclude_manifests=()):
    output = Path(output).resolve()
    if phase not in ('dev', 'confirmation'):
        raise ValueError('Explicit dev or confirmation phase required')
    if max_seconds <= 0 or not math.isfinite(max_seconds) or max_decisions < 1:
        raise ValueError('Positive finite per-match limits required')
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != '.evaluation.lock' for p in output.iterdir()):
        raise ValueError('New evaluations require an empty directory; resume an existing manifest')
    exclusions = []
    for path in exclude_manifests:
        other = read_json(path)
        if other['game'] == game and set(seeds) & set(other['seeds']):
            raise ValueError('Evaluation seeds overlap an excluded development/training manifest')
        exclusions.append(dict(path=str(Path(path).resolve()), sha256=file_hash(path)))
    bots = freeze_bots(bot_file, output, game)
    cls = game_class(game)
    manifest = dict(format=FORMAT, game=game, config=cls.config_id, rules_version=cls.rules_version,
                    phase=phase, seeds=list(seeds), bots=bots,
                    bots_input=dict(path=str(Path(bot_file).resolve()), sha256=file_hash(bot_file)),
                    settings=dict(max_seconds=float(max_seconds), max_decisions=int(max_decisions),
                                  record=bool(record), validate=bool(validate)),
                    schedule=schedule_matches(game, bots, seeds),
                    excluded_manifests=exclusions, code=archive_source(output),
                    created_unix=time.time(), selection='none; frozen candidates only')
    manifest['manifest_sha256'] = canonical_hash(manifest)
    write_json(output/'manifest.json', manifest)
    return manifest


def load_manifest(output, verify_code=True):
    output = Path(output).resolve()
    manifest = read_json(output/'manifest.json')
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported evaluation format')
    original_hash = manifest['manifest_sha256']
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if canonical_hash(unsigned) != original_hash:
        raise ValueError('Frozen manifest identity changed')
    if verify_code and code_identity()['content_sha256'] != manifest['code']['content_sha256']:
        raise ValueError('Resume requires the same code/data snapshot; restore source.tar.gz')
    if file_hash(output/'source.tar.gz') != manifest['code']['archive_sha256']:
        raise ValueError('Source archive changed')
    for bot in manifest['bots'].values():
        if bot['model']:
            model = bot['model']
            path = output/model['relative_path']
            if file_hash(path) != model['sha256']:
                raise ValueError('Frozen model identity changed')
            # Allows an entire evaluation directory to move, without changing the
            # content-based policy identity or reading its old best.json again.
            bot['spec']['path'] = str(path)
    return manifest


def worker_init():
    if hasattr(os, 'setsid'):
        os.setsid()  # Our worker and any compiler descendants form an owned group.
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'


def frozen_match_worker(job):
    # Import the common runner only after worker initialization constrains BLAS.
    began, cpu = time.monotonic(), time.process_time()
    from .train import match_worker
    try:
        for spec in job['specs']:
            if spec['kind'] in ('neural', 'harmonies_rl') and file_hash(spec['path']) != spec['sha256']:
                raise ValueError('Frozen model checksum changed before match')
        result = match_worker(job)
        for spec in job['specs']:
            if spec['kind'] in ('neural', 'harmonies_rl') and file_hash(spec['path']) != spec['sha256']:
                raise ValueError('Frozen model checksum changed during match')
    except Exception as error:
        result = dict(game=job['game_id'], seed=job['seed'], first_player=job['first_player'],
                      policies=job['specs'], status='error', rewards=None,
                      seconds=time.monotonic()-began, error=repr(error))
    result.update(worker_wall_seconds=time.monotonic()-began, worker_cpu_seconds=time.process_time()-cpu)
    return result


def latest_attempts(output):
    latest, attempts, identities = {}, [], set()
    for path in sorted((Path(output)/'attempts').glob('*.json')):
        record = read_json(path)
        identity = (record['match_id'], record['attempt'])
        if (record.get('filename') != path.name or type(record['attempt']) is not int
                or record['attempt'] < 1 or identity in identities):
            raise ValueError('Invalid or duplicate attempt identity')
        identities.add(identity)
        attempts.append(record)
        key = record['match_id']
        if key not in latest or record['attempt'] > latest[key]['attempt']:
            latest[key] = record
    return latest, attempts


def scheduled_results(manifest, latest):
    return [dict(row, result=deepcopy(latest[row['match_id']]['result']) if row['match_id'] in latest
                 else dict(status='not_started', rewards=None, seconds=0.))
            for row in manifest['schedule']]


def win_points(result, seat):
    if result['status'] != 'complete':
        return None
    rewards = result.get('rewards')
    if not isinstance(rewards, list) or len(rewards) != 2 or any(
            v not in (-1., 0., 1.) for v in rewards) or sum(rewards) != 0:
        raise ValueError('A completed match requires valid zero-sum terminal rewards')
    return .5+.5*rewards[seat]


def percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def interval(values):
    return [percentile(values, .025), percentile(values, .975)] if values else None


def pair_statistics(rows, bot, bootstrap_samples=2000, bootstrap_seed=160):
    """Resample environment-seed clusters, never individual swapped-seat games."""
    if bootstrap_samples < 1 or not rows:
        raise ValueError('Nonempty paired rows and positive bootstrap sample count required')
    groups = defaultdict(list)
    for row in rows:
        groups[row['seed']].append(row)
    stats, points, scores = [], [], []
    missing_scores = 0
    wins = draws = losses = 0
    for seed in sorted(groups):
        pair = groups[seed]
        if len(pair) != 2 or {r['orientation'] for r in pair} != {0, 1}:
            raise ValueError('Each environment seed requires both scheduled seat orientations')
        group_points, complete = 0., 0
        for row in pair:
            seat = row['seat_names'].index(bot)
            value = win_points(row['result'], seat)
            if value is None:
                continue
            complete += 1
            group_points += value
            points.append(value)
            wins += value == 1
            draws += value == .5
            losses += value == 0
            raw = row['result'].get('outcome', {}).get('scores')
            score = raw[seat] if isinstance(raw, list) and len(raw) == 2 else None
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                scores.append(float(score))
            else:
                missing_scores += 1
        stats.append((group_points, complete, 2))
    rng = random.Random(bootstrap_seed)
    lower_bootstrap, completed_bootstrap, paired_bootstrap = [], [], []
    fully_paired = [p/2 for p, c, _ in stats if c == 2]
    for _ in range(bootstrap_samples):
        sample = [stats[rng.randrange(len(stats))] for _ in stats]
        total_points = sum(s[0] for s in sample)
        total_complete = sum(s[1] for s in sample)
        lower_bootstrap.append(total_points/sum(s[2] for s in sample))
        if total_complete:
            completed_bootstrap.append(total_points/total_complete)
        if fully_paired:
            paired_bootstrap.append(sum(rng.choice(fully_paired) for _ in fully_paired)/len(fully_paired))
    scheduled, completed = len(rows), len(points)
    total = sum(points)
    return dict(scheduled=scheduled, completed=completed, incomplete=scheduled-completed,
                wins=wins, draws=draws, losses=losses, win_points=total,
                all_scheduled_win_rate=total/scheduled,
                all_scheduled_win_rate_bounds=[total/scheduled, (total+scheduled-completed)/scheduled],
                completed_win_rate=total/completed if completed else None,
                paired_complete_seeds=len(fully_paired),
                paired_complete_seed_win_rate=sum(fully_paired)/len(fully_paired) if fully_paired else None,
                score=dict(mean=sum(scores)/completed if completed and not missing_scores else None,
                           observed_count=len(scores), missing_count=missing_scores),
                bootstrap=dict(method='percentile; resample whole environment seeds with both seats',
                    samples=bootstrap_samples, seed=bootstrap_seed, independent_seeds=len(stats),
                    all_scheduled_win_rate_ci95=interval(lower_bootstrap),
                    completed_win_rate_ci95=interval(completed_bootstrap),
                    paired_complete_seed_win_rate_ci95=interval(paired_bootstrap),
                    undefined_completed_resamples=bootstrap_samples-len(completed_bootstrap)))


def build_summary(manifest, rows, attempts=(), bootstrap_samples=2000, bootstrap_seed=160):
    pairs = defaultdict(list)
    for row in rows:
        pairs[tuple(row['pair'])].append(row)
    reports = []
    for pair in sorted(pairs):
        selected = pairs[pair]
        reports.append(dict(bots=list(pair), scheduled=len(selected),
            statuses=dict(sorted(Counter(r['result']['status'] for r in selected).items())),
            policies={name: pair_statistics(selected, name, bootstrap_samples, bootstrap_seed)
                      for name in pair},
            match_seconds=sum(r['result'].get('seconds') or 0. for r in selected)))
    return dict(format=FORMAT, game=manifest['game'], phase=manifest['phase'],
                manifest_sha256=manifest['manifest_sha256'], selection_performed=False,
                scheduled=len(rows), completed=sum(r['result']['status'] == 'complete' for r in rows),
                statuses=dict(sorted(Counter(r['result']['status'] for r in rows).items())),
                pair_reports=reports, attempts=len(attempts),
                attempt_worker_wall_seconds=sum(a['result'].get('worker_wall_seconds') or 0. for a in attempts),
                measured_worker_cpu_seconds=sum(a['result'].get('worker_cpu_seconds') or 0. for a in attempts),
                unmeasured_cpu_attempts=sum('worker_cpu_seconds' not in a['result'] for a in attempts),
                notes=['All scheduled denominators include unstarted/error/unresolved/truncated matches.',
                       'Incomplete matches have unknown outcomes, not losses; the scheduled rate is a lower bound.',
                       'Score means use terminal scores only; null race scores remain null.',
                       'Intervals are descriptive seed-cluster bootstrap intervals, not multiple-comparison corrections.'])


def save_report(output, manifest, bootstrap_samples=2000, bootstrap_seed=160, *,
                attempt_state=None):
    """Rebuild from disk, or use the writer's already-persisted in-memory records.

    The optional pair is (latest, attempts), owned by this invocation under its
    evaluation lock. Offline callers keep the complete disk-read behavior.
    """
    latest, attempts = latest_attempts(output) if attempt_state is None else attempt_state
    # Match the disk scanner's filename order, including floating-point resource
    # sums, even when new attempts were submitted in a different order.
    attempts = sorted(attempts, key=lambda attempt: attempt['filename'])
    rows = scheduled_results(manifest, latest)
    summary = build_summary(manifest, rows, attempts, bootstrap_samples, bootstrap_seed)
    invocations = [read_json(p) for p in Path(output).glob('invocation_*.json')]
    summary['invocations'] = len(invocations)
    summary['invocation_wall_seconds'] = sum(r.get('wall_seconds', 0.) for r in invocations)
    write_json(Path(output)/'games.json', rows)
    write_json(Path(output)/'summary.json', summary)
    return summary


@contextmanager
def evaluation_lock(output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.evaluation.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This evaluation directory already has a running writer') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def terminate_pool(pool):
    # Python <=3.13 has no public terminate_workers(). Kill only our own pool;
    # a per-game timer cannot interrupt a single stuck environment/policy call.
    processes = list((getattr(pool, '_processes', {}) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    groups = []
    for process in processes:
        if process.is_alive():
            try:
                if hasattr(os, 'killpg') and os.getpgid(process.pid) == process.pid:
                    groups.append(process.pid)
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    end = time.monotonic()+2.
    for process in processes:
        process.join(timeout=max(0., end-time.monotonic()))
        if process.is_alive():
            process.kill()
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.join(timeout=.05)


def run_evaluation(output, workers=1, deadline_unix=None, bootstrap_samples=2000,
                   bootstrap_seed=160, verify_code=True):
    if workers < 1:
        raise ValueError('Positive worker count required')
    if deadline_unix is not None and not math.isfinite(deadline_unix):
        raise ValueError('Deadline must be a finite Unix timestamp')
    output = Path(output).resolve()
    with evaluation_lock(output):
        manifest = load_manifest(output, verify_code)
        began = time.time()
        (output/'attempts').mkdir(exist_ok=True)
        latest, attempts = latest_attempts(output)
        scheduled_ids = {r['match_id'] for r in manifest['schedule']}
        if any(a.get('manifest_sha256') != manifest['manifest_sha256']
               or a['match_id'] not in scheduled_ids for a in attempts):
            raise ValueError('Attempt belongs to another frozen evaluation')

        def persist_attempt(attempt, **changes):
            # The mirror advances only after the atomic file replacement succeeds.
            # latest/attempts/running share the same record object, including old
            # attempts whose interrupted result must survive a subsequent retry.
            write_json(output/'attempts'/attempt['filename'], dict(attempt, **changes))
            attempt.update(changes)

        # A prior writer's abrupt exit cannot silently erase its running attempt.
        for attempt in latest.values():
            if attempt['result']['status'] == 'running':
                persist_attempt(attempt, result=dict(status='interrupted', rewards=None, seconds=None,
                    error='Previous invocation exited without recording an outcome'))
        pending = [r for r in manifest['schedule'] if r['match_id'] not in latest
                   or latest[r['match_id']]['result']['status'] in RETRYABLE]
        invocation = dict(started_unix=began, deadline_unix=deadline_unix, workers=workers,
                          manifest_sha256=manifest['manifest_sha256'], pending_at_start=len(pending))
        invocation_path = output/f'invocation_{time.time_ns()}.json'
        write_json(invocation_path, invocation)
        pool = None
        running = {}
        stopped = None
        active_error = None
        try:
            if pending and not STOP and (deadline_unix is None or time.time() < deadline_unix):
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                           initializer=worker_init)
            while pending or running:
                # Harvest already-finished work before considering a deadline.
                for future in [f for f in running if f.done()]:
                    attempt, started = running[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = dict(status='error', rewards=None, seconds=time.time()-started,
                                      error=repr(error))
                    persist_attempt(attempt, result=result, finished_unix=time.time())
                    running.pop(future)
                if STOP or (deadline_unix is not None and time.time() >= deadline_unix):
                    stopped = 'interrupted' if STOP else 'deadline'
                    break
                while (pending and len(running) < workers and not STOP
                       and (deadline_unix is None or time.time() < deadline_unix)):
                    row = pending.pop(0)
                    previous = latest.get(row['match_id'])
                    number = previous['attempt']+1 if previous else 1
                    filename = f"{row['match_id']}_{number:04d}.json"
                    started = time.time()
                    attempt = dict(match_id=row['match_id'], attempt=number, filename=filename,
                        manifest_sha256=manifest['manifest_sha256'],
                        started_unix=started, result=dict(status='running', rewards=None, seconds=None))
                    persist_attempt(attempt)
                    attempts.append(attempt)
                    latest[row['match_id']] = attempt
                    specs = [manifest['bots'][name]['spec'] for name in row['seat_names']]
                    job = dict(game_id=manifest['game'], specs=specs, seed=row['seed'],
                               first_player=row['first_player'], **manifest['settings'])
                    try:
                        future = pool.submit(frozen_match_worker, job)
                    except Exception as error:
                        persist_attempt(attempt, finished_unix=time.time(), result=dict(status='error',
                            rewards=None, seconds=0., error=repr(error)))
                        raise
                    running[future] = (attempt, started)
                if running:
                    timeout = min(.5, max(0., deadline_unix-time.time())) if deadline_unix else .5
                    wait(running, timeout=timeout, return_when=FIRST_COMPLETED)
                elif not pending:
                    break
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                try:
                    if running:
                        for attempt, started in running.values():
                            persist_attempt(attempt, finished_unix=time.time(), result=dict(status=stopped or 'interrupted',
                                rewards=None, seconds=None, worker_wall_seconds=time.time()-started,
                                error='Owned worker stopped before a terminal result was recorded'))
                finally:
                    # A failed interruption write must never orphan owned workers.
                    if pool is not None:
                        if running:
                            terminate_pool(pool)
                        else:
                            pool.shutdown(wait=True)
            except BaseException as cleanup_error:
                if active_error is not None:
                    raise active_error from cleanup_error
                raise
            # Failed cleanup propagates above: no finished invocation or report
            # may claim that an unpersisted outcome was successfully recorded.
            invocation.update(finished_unix=time.time(), wall_seconds=time.time()-began,
                              stop_reason=stopped)
            write_json(invocation_path, invocation)
            summary = save_report(output, manifest, bootstrap_samples, bootstrap_seed,
                                  attempt_state=(latest, attempts))
        return summary


def stop_requested(*_):
    global STOP
    STOP = True


def parse_seeds(text, start, count):
    if text is not None:
        seeds = [int(x.strip()) for x in text.split(',') if x.strip()]
    else:
        if count < 1:
            raise ValueError('Positive seed count required')
        seeds = list(range(start, start+count))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Specify nonempty unique environment seeds')
    return seeds


def main(argv=None):
    global STOP
    STOP = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True)
    parser.add_argument('--bots', type=Path, required=True, help='JSON name -> spec object or spec JSON path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', required=True, choices=['dev', 'confirmation'])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seeds', help='Comma-separated environment seeds')
    group.add_argument('--seed-start', type=int)
    parser.add_argument('--num-seeds', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--wall-seconds', type=float, default=3600.)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--max-match-seconds', type=float, default=120.)
    parser.add_argument('--max-decisions', type=int, default=10000)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=160)
    parser.add_argument('--exclude-manifest', action='append', default=[], type=Path)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.wall_seconds < 0 or not math.isfinite(args.wall_seconds):
        parser.error('wall-seconds must be finite and nonnegative')
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    if args.resume:
        manifest = load_manifest(args.output)
        expected = dict(max_seconds=args.max_match_seconds, max_decisions=args.max_decisions,
                        record=args.record, validate=args.validate)
        if (manifest['game'] != args.game or manifest['phase'] != args.phase
                or manifest['seeds'] != seeds or manifest['settings'] != expected
                or manifest['bots_input']['sha256'] != file_hash(args.bots)):
            raise ValueError('Resume must retain the original game/phase/seeds/settings/bots input')
    else:
        with evaluation_lock(args.output):
            prepare_evaluation(args.game, args.bots, args.output, seeds, args.phase,
                               args.max_match_seconds, args.max_decisions, args.record,
                               args.validate, args.exclude_manifest)
    deadline = time.time()+args.wall_seconds
    if args.deadline_unix is not None:
        deadline = min(deadline, args.deadline_unix)
    summary = run_evaluation(args.output, args.workers, deadline,
                             args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
