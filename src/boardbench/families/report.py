"""Read-only progress, frozen-league evidence and allocated-resource reporting.

No processes are probed, models selected, results changed or jobs controlled.
Run ``python -m boardbench.families.report --help`` for the local JSON interface.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time

from .evaluate import FORMAT, build_summary, canonical_hash, scheduled_results

RESULT_FILE = re.compile(r'(bootstrap|confirmation|g\d+_(population|gate)|rollout_\d+|gate_\d+)\.json')
FULL_JSON_LIMIT = 256 * 1024
READ_THREADS = min(4, max(1, (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)//2))
RESOURCE_NOTE = (
    'Allocated CPU core-hours / GPU card-hours are interval upper bounds, not measured busy usage. '
    'Intervals run from recorded launch to exit-file mtime, or report time without an exit. '
    'Missing/replaced launch history and changed allocations on resume cannot be reconstructed; '
    'these are not guaranteed lifetime bounds. Shared resource intervals are merged in totals.'
)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def count(value):
    return value if type(value) is int and value >= 0 else None


def read_json(path, issues, expected=dict):
    """Diagnostics never echo file contents, exception text, commands or credentials."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, expected):
            raise ValueError('schema')
        return data
    except (OSError, ValueError, TypeError):
        issues.append(dict(code='unreadable_or_invalid_json', file=path.name))
        return None


def read_json_files(paths):
    def load(path):
        issues = []
        return path, read_json(path, issues), issues
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        yield from pool.map(load, paths)


def header_events(chunk):
    """C-level literal searches avoid a regex check at every trajectory byte."""
    chunk = b'\n'+chunk; events = []
    for token, boundary in ((b'\n  {\n', b'{'), (b'\n  },\n', b'},'), (b'\n  }\n', b'}'),
                            (b'\n  {},\n', b'{},'), (b'\n  {}\n', b'{}')):
        start = 0
        while (position := chunk.find(token, start)) >= 0:
            events.append((position, boundary, None)); start = position+len(token)-1
    token = b'\n    "status": '; start = 0
    while (position := chunk.find(token, start)) >= 0:
        end = chunk.find(b'\n', position+1)
        encoded = chunk[position+len(token):end].removesuffix(b',')
        events.append((position, None, encoded)); start = end
    return sorted(events)


def training_statuses(path, issues, full_json_limit=FULL_JSON_LIMIT):
    """Read row headers without materializing large, pretty-printed trajectories.

    Large files use the exact indent=2 format of play.write_json. This validates
    outer list/row boundaries and status headers, not nested trajectory JSON.
    Reads are bounded to 4 MiB; very long non-header lines are skipped safely.
    """
    try:
        if path.stat().st_size <= full_json_limit:
            rows = read_json(path, issues, list)
            if rows is None:
                return None
            valid = [r['status'] for r in rows if isinstance(r, dict) and isinstance(r.get('status'), str)]
            return Counter(valid), len(rows)-len(valid), 'full_json'
        statuses = Counter(); invalid = 0; opened = False; status = None
        with path.open('rb') as handle:
            if handle.read(2) != b'[\n':
                raise ValueError('unsupported large format')
            handle.seek(-2, 2)
            if handle.read() not in (b'\n]', b']\n'):
                raise ValueError('unfinished list')
            handle.seek(0); carry = b''; discard_line = False
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                if discard_line:
                    newline = chunk.find(b'\n')
                    if newline < 0:
                        continue
                    chunk = chunk[newline+1:]; discard_line = False
                chunk = carry+chunk; end = chunk.rfind(b'\n')
                if end < 0:
                    carry = chunk
                    if len(carry) > 4096:
                        carry = b''; discard_line = True
                    continue
                carry = chunk[end+1:]
                if len(carry) > 4096:
                    carry = b''; discard_line = True
                for _, boundary, encoded in header_events(chunk[:end+1]):
                    if boundary == b'{':
                        if opened:
                            raise ValueError('nested row boundary')
                        opened = True; status = None
                    elif boundary in (b'}', b'},'):
                        if not opened:
                            raise ValueError('unmatched row boundary')
                        if status is None:
                            invalid += 1
                        else:
                            statuses[status] += 1
                        opened = False
                    elif boundary in (b'{}', b'{},'):
                        if opened:
                            raise ValueError('nested empty row')
                        invalid += 1
                    elif encoded is not None:
                        if not opened or status is not None:
                            raise ValueError('invalid row status header')
                        value = json.loads(encoded)
                        status = value if isinstance(value, str) else None
            if opened or not statuses and not invalid:
                raise ValueError('unrecognized large result rows')
        return statuses, invalid, 'pretty_json_row_headers'
    except (OSError, ValueError, TypeError):
        issues.append(dict(code='unreadable_or_unsupported_result_headers', file=path.name))
        return None


def written_training(directory, issues):
    statuses, stages = Counter(), Counter()
    files = invalid_rows = 0; modes = Counter()
    paths = sorted(p for p in directory.glob('*.json') if RESULT_FILE.fullmatch(p.name))
    def scan(path):
        local_issues = []
        return path, training_statuses(path, local_issues), local_issues
    # Shared filesystem metadata/open latency dominates thousands of result files.
    # A small read-only pool keeps memory bounded and never touches training jobs.
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        for path, result, local_issues in pool.map(scan, paths):
            issues.extend(local_issues)
            if result is None:
                continue
            counts, bad, mode = result; invalid_rows += bad; statuses.update(counts); modes[mode] += 1
            files += 1
            stage = ('population' if '_population' in path.stem else 'rollout' if path.stem.startswith('rollout_')
                     else 'gate' if 'gate' in path.stem else path.stem)
            stages[stage] += sum(counts.values())
    if invalid_rows:
        issues.append(dict(code='invalid_match_rows', count=invalid_rows))
    return dict(files=files, recorded=sum(statuses.values()), completed=statuses['complete'],
                incomplete=sum(v for k, v in statuses.items() if k != 'complete'),
                statuses=dict(sorted(statuses.items())), stages=dict(sorted(stages.items())),
                file_read_modes=dict(modes),
                count_basis='Actual readable result rows; never inferred from update/population settings.',
                large_file_validation='Files above 256 KiB: streaming indent=2 row headers only; nested trajectories are not revalidated.',
                history_scope='Current files only; overwritten batches or missing files cannot be recovered.')


def league(directory, issues, bootstrap_samples):
    """Recompute a read-only view from immutable attempts; preserve the frozen schedule."""
    manifest = read_json(directory/'manifest.json', issues)
    if manifest is None:
        return None
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    try:
        valid = manifest.get('format') == FORMAT and canonical_hash(unsigned) == manifest['manifest_sha256']
        schedule = manifest['schedule']
        ids = {r['match_id'] for r in schedule}
        valid = (valid and isinstance(schedule, list) and bool(schedule) and len(ids) == len(schedule)
                 and isinstance(manifest.get('bots'), dict)
                 and all(isinstance(b, dict) and isinstance(b.get('spec'), dict)
                         and (b.get('model') is None or isinstance(b.get('model'), dict))
                         for b in manifest['bots'].values()))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        issues.append(dict(code='invalid_frozen_manifest', file='manifest.json'))
        return None
    latest, attempts, identities = {}, [], set()
    for path, record, local_issues in read_json_files(sorted((directory/'attempts').glob('*.json'))):
        issues.extend(local_issues)
        if record is None:
            continue
        key = record.get('match_id'); attempt = record.get('attempt')
        if (not isinstance(key, str) or key not in ids or type(attempt) is not int or attempt < 1
                or record.get('filename') != path.name or (key, attempt) in identities
                or not isinstance(record.get('result'), dict)
                or not isinstance(record['result'].get('status'), str)):
            issues.append(dict(code='invalid_attempt', file=path.name))
            continue
        # Retain only fields needed for the statistical view; recorded trajectories,
        # exception bodies and private policy specs never accumulate in the report.
        result = record['result']
        record = dict(match_id=key, attempt=attempt, filename=path.name,
                      result={k: result[k] for k in ('status', 'rewards', 'seconds', 'worker_wall_seconds',
                              'worker_cpu_seconds') if k in result})
        record['result']['outcome'] = dict(scores=result.get('outcome', {}).get('scores')) if isinstance(result.get('outcome'), dict) else {}
        identities.add((key, attempt)); attempts.append(record)
        if key not in latest or attempt > latest[key]['attempt']:
            latest[key] = record
    statuses = Counter(a['result']['status'] for a in attempts)
    try:
        rows = scheduled_results(manifest, latest)
    except (KeyError, TypeError):
        issues.append(dict(code='invalid_frozen_schedule', file='manifest.json'))
        return None
    current_statuses = Counter(r['result']['status'] for r in rows)
    written = dict(recorded_attempts=len(attempts), attempt_statuses=dict(sorted(statuses.items())),
                   unique_matches_with_attempts=len(latest), latest_completed=current_statuses['complete'],
                   not_started=current_statuses['not_started'])
    saved = read_json(directory/'summary.json', issues) if (directory/'summary.json').exists() else None
    saved_matches = bool(saved and saved.get('manifest_sha256') == manifest['manifest_sha256']
                         and saved.get('attempts') == len(attempts)
                         and saved.get('scheduled') == len(schedule)
                         and saved.get('statuses') == dict(current_statuses))
    # Always rebuild from visible attempts so equal counts do not conceal changed outcomes.
    # This neither writes the original summary nor performs model selection.
    try:
        summary = build_summary(manifest, rows, attempts, bootstrap_samples=bootstrap_samples)
        pair_reports = summary['pair_reports']
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        issues.append(dict(code='invalid_league_results', file='attempts'))
        pair_reports = None
    bots = {}
    for name, bot in manifest.get('bots', {}).items():
        model = bot.get('model') or {}
        bots[name] = dict(kind=bot.get('spec', {}).get('kind'), identity_sha256=bot.get('identity_sha256'),
                          model_sha256=model.get('sha256'))
    return dict(phase=manifest.get('phase'), manifest_sha256=manifest['manifest_sha256'],
                config=manifest.get('config'), rules_version=manifest.get('rules_version'), bots=bots,
                scheduled=len(schedule), completed=current_statuses['complete'],
                statuses=dict(sorted(current_statuses.items())), written=written,
                pair_reports=pair_reports, saved_summary_present=saved is not None,
                saved_summary_count_check=saved_matches,
                selection_performed=False, provenance='Frozen manifest and latest readable attempt per match.',
                confidence_note='95% descriptive environment-seed cluster bootstrap; both seats stay together. '
                    'Scheduled rates are lower bounds with incomplete outcomes unknown; no multiple-comparison correction.',
                model_integrity_note='Recorded frozen identities only; report does not load or rehash model binaries.',
                invocations=len(list(directory.glob('invocation_*.json'))))


def allocation(launch, job, now, issues):
    """Return a publishable bound and private keys used only for interval unions."""
    start = launch.get('launched_unix')
    end = now; endpoint = 'report_time'; exit_code = None; exit_mtime = None
    exit_file = job/'exit_code'
    if exit_file.exists():
        try:
            exit_code = int(exit_file.read_text().strip())
            exit_mtime = exit_file.stat().st_mtime
            end = exit_mtime; endpoint = 'exit_file_mtime'
            if end > now:
                # A long read may observe an exit written after the report's
                # reference time. Bound allocations at that reference time.
                end = now; endpoint = 'report_time_before_observed_exit'
        except (OSError, ValueError):
            issues.append(dict(code='invalid_exit_code', file='exit_code'))
    cpus = launch.get('cpu_ids')
    cpus = list(dict.fromkeys(cpus)) if isinstance(cpus, list) and all(type(x) is int and x >= 0 for x in cpus) else None
    gpu = launch.get('gpu')
    gpus = [gpu] if type(gpu) is int and gpu >= 0 else []
    if launch.get('kind') == 'ppo' and not gpus:
        issues.append(dict(code='missing_gpu_allocation'))
        gpus = None
    valid_time = number(start) and number(end) and start <= end <= now + 1
    if number(start) and exit_mtime is not None and exit_mtime < start:
        issues.append(dict(code='exit_predates_launch', file='exit_code'))
    if not valid_time:
        issues.append(dict(code='invalid_allocation_interval'))
    if cpus is None:
        issues.append(dict(code='missing_cpu_allocation'))
    duration = end-start if valid_time else None
    result = dict(recorded_launch_unix=start if number(start) else None,
                  observed_exit_unix=exit_mtime,
                  endpoint_unix=end if number(end) else None, endpoint=endpoint,
                  allocated_seconds_upper_bound=duration, allocated_cpu_count=len(cpus) if cpus is not None else None,
                  allocated_gpu_count=len(gpus) if gpus is not None else None,
                  allocated_cpu_core_hours_upper_bound=duration*len(cpus)/3600 if valid_time and cpus is not None else None,
                  allocated_gpu_card_hours_upper_bound=duration*len(gpus)/3600 if valid_time and gpus is not None else None,
                  resume_history='Not reconstructed; covers this recorded launch interval only.')
    private = dict(host=launch.get('host'), cpus=cpus, gpus=gpus, start=start, end=end, valid=valid_time)
    return result, private, exit_code


def union_seconds(intervals):
    total = 0.; right = None
    for start, end in sorted(intervals):
        total += max(0., end-max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def resource_totals(tasks, allocations):
    intervals = {'cpu': defaultdict(list), 'gpu': defaultdict(list)}
    unknown = Counter()
    for item in allocations:
        for kind in intervals:
            ids = item['cpus' if kind == 'cpu' else 'gpus']
            if not item['valid'] or not isinstance(item['host'], str) or not item['host'] or ids is None:
                unknown[kind] += 1
                continue
            for identifier in ids:
                intervals[kind][(item['host'], identifier)].append((item['start'], item['end']))
    return dict(
        task_allocated_cpu_core_hours_upper_bound=sum(t['resources'].get('allocated_cpu_core_hours_upper_bound') or 0 for t in tasks),
        task_allocated_gpu_card_hours_upper_bound=sum(t['resources'].get('allocated_gpu_card_hours_upper_bound') or 0 for t in tasks),
        unique_allocated_cpu_core_hours_upper_bound=sum(union_seconds(v) for v in intervals['cpu'].values())/3600,
        unique_allocated_gpu_card_hours_upper_bound=sum(union_seconds(v) for v in intervals['gpu'].values())/3600,
        tasks_missing_cpu_interval=unknown['cpu'], tasks_missing_gpu_interval=unknown['gpu'],
        tasks_without_launch=sum(not t['has_launch'] for t in tasks), note=RESOURCE_NOTE)


def build_report(root, now=None, include_host_aliases=False, bootstrap_samples=2000, skip_training_records=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Bots root must be an existing directory')
    now = time.time() if now is None else float(now)
    if not math.isfinite(now) or bootstrap_samples < 1:
        raise ValueError('Finite report time and positive bootstrap count required')
    candidates, claimed = [], set()
    for path in sorted(root.glob('*/*/launch.json')):
        issues = []; launch = read_json(path, issues) or {}; job = path.parent
        # An evaluation launcher lives in a sibling *_job directory. Do not follow
        # absolute paths from the manifest (portable runs may have moved).
        data = job
        if launch.get('kind') == 'eval':
            name = Path(str(launch.get('output', ''))).name
            data = job.with_name(name) if name and name not in ('.', '..') else job
        candidates.append((job, data, launch, issues)); claimed.add(data)
    for path in sorted(set(root.glob('*/*/state.json')) | set(root.glob('*/*/manifest.json'))):
        if path.parent not in claimed:
            candidates.append((path.parent, path.parent, {}, [])); claimed.add(path.parent)
    tasks, allocations = [], []
    for job, data, launch, issues in candidates:
        # Refuse symlinks that lead outside the requested experiment root.
        try:
            job.resolve().relative_to(root); data.resolve().relative_to(root)
        except ValueError:
            continue
        state = read_json(data/'state.json', issues) if (data/'state.json').exists() else {}
        state = state or {}
        evaluation = league(data, issues, bootstrap_samples) if (data/'manifest.json').exists() else None
        if launch:
            resources, private, exit_code = allocation(launch, job, now, issues); allocations.append(private)
        else:
            resources = {}; exit_code = None
        training = written_training(data, issues) if evaluation is None and not skip_training_records else None
        phase = state.get('phase')
        if phase is not None and not isinstance(phase, str):
            issues.append(dict(code='invalid_state_phase', file='state.json')); phase = None
        if exit_code is not None:
            status = 'completed' if exit_code == 0 else 'abnormal'
        elif launch:
            status = 'activity_unverified'
        elif phase in ('complete', 'completed') or evaluation and evaluation['completed'] == evaluation['scheduled']:
            status = 'artifacts_complete'
        else:
            status = 'unknown'
        if phase and (phase.startswith('blocked') or phase in ('failed', 'error', 'stopped', 'stopping')):
            status = 'abnormal'
        if evaluation and status == 'completed' and evaluation['completed'] != evaluation['scheduled']:
            status = 'abnormal'; issues.append(dict(code='evaluation_incomplete_after_exit'))
        if evaluation and any(n for key, n in evaluation['statuses'].items()
                              if key not in ('complete', 'not_started', 'running')):
            issues.append(dict(code='recorded_noncomplete_evaluation'))
        deadline = launch.get('deadline')
        if exit_code is None and status == 'activity_unverified' and number(deadline) and now > deadline:
            issues.append(dict(code='deadline_passed_without_exit_record'))
        updated = state.get('updated_unix'); exited = resources.get('observed_exit_unix')
        if number(updated) and number(exited) and updated > exited+1:
            issues.append(dict(code='state_updated_after_exit_record', file='exit_code'))
        if issues:
            status = 'abnormal'
        task = dict(task=str(job.relative_to(root)), data_run=str(data.relative_to(root)),
                    game=data.parent.name, kind=launch.get('kind') or ('eval' if evaluation else 'unknown'),
                    has_launch=bool(launch), status=status, phase=phase, exit_code=exit_code,
                    process_liveness='unverified',
                    status_evidence='exit_record' if exit_code is not None else 'launch_without_exit' if launch else 'artifacts_without_launch',
                    process_liveness_note='No process was probed. A complete state snapshot does not prove that a resumed task has exited.',
                    progress=dict(completed_updates=count(state.get('next_update')),
                                  completed_generations=count(state.get('next_generation')),
                                  state_updated_unix=state.get('updated_unix') if number(state.get('updated_unix')) else None),
                    written_training=training, league=evaluation, resources=resources, issues=issues)
        if include_host_aliases and re.fullmatch(r'[ab]\d+', str(launch.get('host', ''))):
            task['host_alias'] = launch['host']
        tasks.append(task)
    return dict(format='boardbench_family_report_v1', generated_unix=now,
                generated_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                visibility='local_with_host_aliases' if include_host_aliases else 'public_summary',
                read_only=True, selection_performed=False, training_records_scanned=not skip_training_records, tasks=tasks,
                task_statuses=dict(sorted(Counter(t['status'] for t in tasks).items())),
                resources=resource_totals(tasks, allocations),
                notes=['No raw commands, API configuration, host addresses, model paths or tracebacks are copied.',
                       'Snapshot is not atomic across concurrently written files; invalid/partial files are flagged, not counted.',
                       'Training result rows and frozen league attempts are different populations; no combined win rate is computed.',
                       'Public JSON still describes experiments; review destination and do not commit raw runtime files.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/v160/bots'))
    parser.add_argument('--output', type=Path, help='New report path outside the bots root; omit to print JSON.')
    parser.add_argument('--include-host-aliases', action='store_true', help='Local report only: include validated a/b node aliases.')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--skip-training-records', action='store_true',
                        help='Quick progress/league view; training game counts are null, never estimated.')
    args = parser.parse_args(argv)
    if args.output:
        try:
            args.output.resolve().relative_to(args.root.resolve())
        except ValueError:
            pass
        else:
            parser.error('Report output must be outside the experiment bots root')
        if args.output.exists():
            parser.error('Refusing to overwrite an existing file; use a new timestamped report path')
    report = build_report(args.root, include_host_aliases=args.include_host_aliases,
                          bootstrap_samples=args.bootstrap_samples, skip_training_records=args.skip_training_records)
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
