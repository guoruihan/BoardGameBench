#!/usr/bin/env python3
"""Read-only audit of V175 direct-play records; no API/adapter invocation.

Emits only aggregate counters and public game/job identities. Private adapter
configuration and prices are not read; response text and raw paths are not emitted.
Token fields deliberately use the three archived adapters' distinct schemas.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path


TOKEN_FIELDS = (
    'input_uncached', 'input_cache_read', 'input_cache_write', 'input_total',
    'output_visible', 'output_reasoning', 'output_total', 'total_tokens',
)
SUM_FIELDS = TOKEN_FIELDS + (
    'requests', 'responses', 'cli_ok', 'cli_failed', 'usage_records',
    'responses_with_usage', 'responses_with_positive_usage', 'response_wall_seconds',
    'result_calls', 'invalid_responses', 'forced_actions', 'agent_decisions',
    'opponent_decisions', 'parent_cpu_seconds', 'match_wall_seconds', 'search_seconds',
)


def normalized_usage(label, usage):
    """Inclusive output/cache fields must not be double-counted."""
    if label == 'opencode':
        uncached = usage.get('input', 0)
        read = usage.get('cache', {}).get('read', 0)
        write = usage.get('cache', {}).get('write', 0)
        visible = usage.get('output', 0)
        reasoning = usage.get('reasoning', 0)
    elif label == 'codex':
        read = usage.get('cached_input_tokens', 0)
        write = usage.get('cache_write_input_tokens', 0)
        # The archived Codex adapter reports cache reads inside input_tokens.
        # No cache writes were observed; do not invent semantics for future ones.
        if write:
            raise ValueError('New Codex cache-write schema needs review')
        uncached = usage.get('input_tokens', 0) - read
        reasoning = usage.get('reasoning_output_tokens', 0)
        visible = usage.get('output_tokens', 0) - reasoning
    elif label == 'claude':
        uncached = usage.get('input_tokens', 0)
        read = usage.get('cache_read_input_tokens', 0)
        write = usage.get('cache_creation_input_tokens', 0)
        reasoning = usage.get('output_tokens_details', {}).get('thinking_tokens', 0)
        visible = usage.get('output_tokens', 0) - reasoning
        # iterations/cache_creation are breakdowns, not additional usage.
    else:
        raise ValueError('Unknown archived adapter label: ' + label)
    values = (uncached, read, write, visible, reasoning)
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('Unexpected token counters: ' + repr(values))
    out = dict(input_uncached=uncached, input_cache_read=read,
               input_cache_write=write, input_total=uncached + read + write,
               output_visible=visible, output_reasoning=reasoning,
               output_total=visible + reasoning, total_tokens=sum(values))
    if label == 'opencode' and 'total' in usage and usage['total'] != out['total_tokens']:
        raise ValueError('OpenCode reported total does not reconcile')
    return out


def read_json(path):
    return json.loads(path.read_text())


def audit_job(root, job):
    directory = root / 'matches' / job['id']
    row = {k: job[k] for k in ('id', 'game', 'opponent', 'seat', 'label')}
    row.update({k: 0 for k in SUM_FIELDS})
    row.update(status='not_started', cpu_recorded=False, wall_recorded=False,
               started_unix=None, ended_unix=None, win=0, draw=0, loss=0)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['job'] != job:
            raise ValueError('Result job identity mismatch: ' + job['id'])
        row['status'] = result['status']
        row['result_calls'] = result.get('calls', 0)
        row['invalid_responses'] = result.get('invalid_responses', 0)
        row['forced_actions'] = result.get('forced_agent_actions', 0)
        decisions = result.get('seat_decisions', [0, 0])
        row['agent_decisions'] = decisions[job['seat']]
        row['opponent_decisions'] = decisions[1 - job['seat']]
        for source, target in [('wall_seconds', 'match_wall_seconds'),
                               ('parent_cpu_seconds', 'parent_cpu_seconds'),
                               ('search_seconds', 'search_seconds')]:
            row[target] = result.get(source) or 0
        row['cpu_recorded'] = result.get('parent_cpu_seconds') is not None
        row['wall_recorded'] = result.get('wall_seconds') is not None
        for key in ('started_unix', 'ended_unix'):
            row[key] = result.get(key)
        if row['status'] == 'complete':
            winners = result['outcome']['winners']
            row['draw'] = int(len(winners) == 2)
            row['win'] = int(len(winners) == 1 and job['seat'] in winners)
            row['loss'] = int(len(winners) == 1 and job['seat'] not in winners)
    elif directory.exists():
        row['status'] = 'missing_result'
    request_names, response_names, raw_dirs = set(), set(), []
    usage_lengths = Counter()
    for path in directory.glob('call_*.request.json'):
        request_names.add(path.name.removesuffix('.request.json'))
    for path in directory.glob('call_*.response.json'):
        response_names.add(path.name.removesuffix('.response.json'))
        response = read_json(path)
        row['responses'] += 1
        row['cli_ok' if response.get('ok') else 'cli_failed'] += 1
        row['response_wall_seconds'] += response.get('wall_seconds', response.get('seconds', 0)) or 0
        usage_list = response.get('usage', [])
        usage_lengths[len(usage_list)] += 1
        row['responses_with_usage'] += bool(usage_list)
        tokens = 0
        for usage in usage_list:
            counts = normalized_usage(job['label'], usage)
            for key, value in counts.items():
                row[key] += value
            tokens += counts['total_tokens']
            row['usage_records'] += 1
        row['responses_with_positive_usage'] += tokens > 0
        if response.get('raw_dir'):
            raw_dirs.append(response['raw_dir'])
    row['requests'] = len(request_names)
    row['missing_responses'] = len(request_names - response_names)
    row['orphan_responses'] = len(response_names - request_names)
    return row, raw_dirs, usage_lengths


def aggregate(rows):
    out = {k: sum(r[k] for r in rows) for k in SUM_FIELDS}
    out.update(scheduled=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
               cpu_records=sum(r['cpu_recorded'] for r in rows),
               wall_records=sum(r['wall_recorded'] for r in rows),
               missing_responses=sum(r['missing_responses'] for r in rows),
               orphan_responses=sum(r['orphan_responses'] for r in rows),
               wins=sum(r['win'] for r in rows), draws=sum(r['draw'] for r in rows),
               losses=sum(r['loss'] for r in rows))
    starts = [r['started_unix'] for r in rows if r['started_unix'] is not None]
    ends = [r['ended_unix'] for r in rows if r['ended_unix'] is not None]
    out['first_started_unix'] = min(starts) if starts else None
    out['last_ended_unix'] = max(ends) if ends else None
    out['observed_span_seconds'] = max(ends) - min(starts) if starts and ends else None
    return out


def audit(root):
    schedule = read_json(root / 'schedule.json')
    jobs = schedule['jobs']
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate scheduled job IDs')
    # Small I/O parallelism only: no simulator, model or remote process is run.
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda job: audit_job(root, job), jobs))
    rows = [r[0] for r in results]
    raw_dirs = [d for r in results for d in r[1]]
    duplicates = len(raw_dirs) - len(set(raw_dirs))
    if duplicates:
        raise ValueError('Duplicate raw call identities need deduplication')
    labels = sorted({j['label'] for j in jobs})
    groups = {label: aggregate([r for r in rows if r['label'] == label]) for label in labels}
    per_game = {label: {game: aggregate([r for r in rows if r['label'] == label and r['game'] == game])
                        for game in sorted({r['game'] for r in rows})} for label in labels}
    completed = {label: aggregate([r for r in rows if r['label'] == label and r['status'] == 'complete'])
                 for label in labels}
    by_opponent = {label: {kind: aggregate([r for r in rows if r['label'] == label and r['opponent'] == kind])
                           for kind in sorted({r['opponent'] for r in rows})} for label in labels}
    usage_lengths = sum((r[2] for r in results), Counter())
    return dict(format_version=1, audited_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='scheduled matches only; excludes smoke/probe calls and unreported provider retries',
                groups=groups, completed_only=completed, per_game=per_game,
                by_opponent=by_opponent, total=aggregate(rows),
                raw_call_identities=len(raw_dirs), duplicate_raw_call_identities=duplicates,
                usage_records_per_response=dict(usage_lengths), jobs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Optional audit output; original run files are unchanged')
    args = parser.parse_args()
    result = audit(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out:
        target = args.out.resolve()
        source = args.root.resolve()
        if target == source or source in target.parents:
            parser.error('--out must be outside the original run directory')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(json.dumps(dict(groups=result['groups'], total=result['total']), ensure_ascii=False, indent=2))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
