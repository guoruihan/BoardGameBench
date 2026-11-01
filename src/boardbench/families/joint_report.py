"""Paired-seed calibration, partial strength order, and common-opponent profiles.

Intervals are exploratory, nominal, and not simultaneous across candidates.
The union with a seed-effective Wilson interval prevents degenerate bootstrap
certainty on tiny all-win/all-loss panels. It is not a coverage guarantee.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import canonical_hash, file_hash
from .joint_pool import verify_pool
from .play import write_json
from .reward_signals import numeric_leaves


def interval(values, seed=174):
    if not values: return None
    x = np.asarray(values, dtype=float); n = len(x); p = float(x.mean())
    rng = np.random.default_rng(seed)
    boot = np.quantile(x[rng.integers(n, size=(4000, n))].mean(1), [.025, .975])
    z = 1.959963984540054; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0., min(float(boot[0]), center-radius)), min(1., max(float(boot[1]), center+radius))]


def pair_summary(jobs, rows, a):
    complete = [r for r in rows if r['status'] == 'complete']
    groups = defaultdict(dict); points = 0.; scores = [[], []]; seconds = [[], []]
    for row in complete:
        seat = row['names'].index(a)
        winners = row['outcome']['winners']
        point = .5 if len(winners) != 1 else float(seat in winners)
        points += point; groups[row['seed']][row['job']['swap']] = point
        for i, s in enumerate((seat, 1-seat)):
            value = row['outcome']['scores'][s]
            if value is not None: scores[i].append(value)
            seconds[i].append(row['policy_seconds'][s])
    paired = [sum(g.values())/2 for g in groups.values() if set(g) == {0, 1}]
    n = len(jobs); unknown = n-len(complete)
    statuses = Counter(r['status'] for r in rows); statuses['not_run'] = n-len(rows)
    return dict(a=a, scheduled=n, recorded=len(rows), complete=len(complete), statuses=dict(statuses),
        win_rate_complete=points/len(complete) if complete else None,
        unknown_outcome_bounds=[points/n, (points+unknown)/n], complete_seed_pairs=len(paired),
        interval=interval(paired), mean_scores=[float(np.mean(s)) if s else None for s in scores],
        mean_policy_seconds=[float(np.mean(s)) if s else None for s in seconds])


def strength_label(summary):
    if not summary or summary['complete'] != summary['scheduled'] or summary['complete_seed_pairs'] < 16:
        return 'insufficient_evidence'
    p = summary['win_rate_complete']; lo, hi = summary['interval']
    if p >= .60 and lo > .5: return 'above_anchor'
    if p <= .40 and hi < .5: return 'below_anchor'
    return 'unresolved_relative_to_anchor'


def budget_violations(rows, members):
    """Attribute decision-budget overruns to the actual search seat only."""
    found = {name: [] for name in members}
    for row in rows:
        for seat, name in enumerate(row['names']):
            spec = members[name]['spec']
            detail = row.get('search', [None, None])[seat]
            if spec['kind'] == 'terminal_search' and detail and detail['spent'] > spec['seconds']:
                found[name].append(dict(job_id=row['job']['id'], spent=detail['spent'], budget=spec['seconds']))
    return found


def style_profiles(rows):
    # Use only the identical named opponent, then average within seed (both seats).
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['status'] != 'complete' or 'legacy' not in row['names']: continue
        for seat, name in enumerate(row['names']):
            if row['names'][1-seat] != 'legacy': continue
            data = numeric_leaves(row['score_breakdown'][seat])
            if row['outcome']['scores'][seat] is not None: data['official_score'] = row['outcome']['scores'][seat]
            count = max(1, row['seat_decisions'][seat])
            data.update({'action_fraction.'+k: v/count for k, v in row['action_counts'][seat].items()})
            data['decisions'] = row['seat_decisions'][seat]
            grouped[name][row['seed']].append((row['job']['swap'], data))
    seed_profiles = {}
    for name, groups in grouped.items():
        seed_profiles[name] = {}
        for seed, samples in groups.items():
            if {s for s, _ in samples} != {0, 1}: continue
            keys = set().union(*(d for _, d in samples))
            seed_profiles[name][seed] = {k: float(np.mean([d.get(k, 0.) for _, d in samples])) for k in keys}
    anchor = seed_profiles.get('legacy', {}); result = {}
    for name, seeds in seed_profiles.items():
        keys = set().union(*(s for s in seeds.values())) if seeds else set()
        means = {k: float(np.mean([s.get(k, 0.) for s in seeds.values()])) for k in sorted(keys)}
        deltas = []; shared = sorted(set(seeds) & set(anchor))
        for key in sorted(keys):
            if key in ('official_score', 'total', 'decisions') or key.startswith('action_fraction.'): continue
            if len(shared) < 16: continue
            x = np.array([seeds[s].get(key, 0.)-anchor[s].get(key, 0.) for s in shared])
            rng = np.random.default_rng(174)
            ci = np.quantile(x[rng.integers(len(x), size=(2000, len(x)))].mean(1), [.025, .975])
            scale = max(1., float(np.std([anchor[s].get(key, 0.) for s in shared])))
            if (ci[0] > 0 or ci[1] < 0) and abs(float(x.mean()))/scale >= .35:
                deltas.append(dict(component=key, delta=float(x.mean()), interval=ci.tolist(),
                                   standardized_delta=float(x.mean())/scale))
        deltas.sort(key=lambda x: abs(x['standardized_delta']), reverse=True)
        result[name] = dict(opponent='legacy', paired_seeds=len(seeds), common_anchor_seeds=len(shared),
            means=means, descriptive_component_shifts=deltas[:4],
            interpretation='Realized behavior against one common opponent; not causal preferences or opponent generalization. Component intervals are nominal and uncorrected.')
    return result


def report(pool, schedule_path, result_dir, output):
    manifest = verify_pool(pool); schedule = json.loads(Path(schedule_path).read_text())
    expected = schedule.pop('schedule_sha256')
    if canonical_hash(schedule) != expected or schedule['pool_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Schedule identity mismatch')
    rows = {}; files = {}
    for job in schedule['jobs']:
        path = Path(result_dir)/(job['id']+'.json')
        if not path.exists(): continue
        row = json.loads(path.read_text()); digest = row.pop('result_sha256')
        if canonical_hash(row) != digest or row['job'] != job or row['pool_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Result identity mismatch: '+job['id'])
        if row.get('trajectory') and file_hash(path.parent/row['trajectory']) != row['trajectory_sha256']:
            raise ValueError('Trajectory identity mismatch')
        if row['status'] == 'complete':
            ids = [manifest['games'][job['game']]['members'][n]['policy_id'] for n in job['names']]
            if row['policy_ids'] != ids: raise ValueError('Played different policies')
        rows[job['id']] = row; files[job['id']] = digest
    pair_jobs = defaultdict(list)
    for job in schedule['jobs']: pair_jobs[(job['game'], *job['pair'])].append(job)
    games = {}
    for game, entry in manifest['games'].items():
        pairs = {}; relative = {}; game_rows = [r for r in rows.values() if r['game'] == game]
        for (g, a, b), jobs in pair_jobs.items():
            if g != game or a == b: continue
            these = [rows[j['id']] for j in jobs if j['id'] in rows]
            key = a+'__'+b
            pairs[key] = dict(b=b, **pair_summary(jobs, these, a))
            if 'legacy' in (a, b):
                other = b if a == 'legacy' else a
                relative[other] = pair_summary(jobs, these, other)
        labels = {n: ('anchor' if n == 'legacy' else strength_label(relative.get(n))) for n in entry['members']}
        violations = budget_violations(game_rows, entry['members'])
        for name, failures in violations.items():
            if failures: labels[name] = 'budget_violation'
        profiles = style_profiles(game_rows)
        above = [n for n, label in labels.items() if label == 'above_anchor']
        economical = min(above, key=lambda n: relative[n]['mean_policy_seconds'][0]) if above else None
        alternatives = [n for n, profile in profiles.items() if n != 'legacy' and not violations[n]
            and relative.get(n, {}).get('complete') == relative.get(n, {}).get('scheduled')
            and (relative.get(n, {}).get('win_rate_complete') or 0.) >= .4
            and any(x['delta'] > 0 for x in profile['descriptive_component_shifts'])]
        games[game] = dict(config=entry['config'], rules_version=entry['rules_version'],
            rules_coverage=entry['rules_coverage'], strength=labels, relative_to_anchor=relative,
            pairs=pairs, styles=profiles, budget_violations=violations,
            three_levels_observed=('above_anchor' in labels.values() and 'below_anchor' in labels.values()),
            roles=dict(anchor='legacy', lower=[n for n, label in labels.items() if label == 'below_anchor'],
                upper=above, economical_above_anchor=economical, style_alternatives=alternatives,
                interpretation='Suggested roles within this frozen pool; economical means fastest among above-anchor members, not stronger than the other members.'))
    statuses = Counter(r['status'] for r in rows.values()); statuses['not_run'] = len(schedule['jobs'])-len(rows)
    summary = dict(format='boardbench_joint_calibration_v1', pool_sha256=manifest['manifest_sha256'],
        schedule_sha256=expected, scheduled=len(schedule['jobs']), recorded=len(rows), statuses=dict(statuses),
        recorded_match_cpu_hours=sum(r.get('cpu_seconds', 0.) for r in rows.values())/3600,
        recorded_match_wall_hours=sum(r.get('wall_seconds', 0.) for r in rows.values())/3600,
        resource_scope='Matches including model loading, excluding process startup and queue; queue worker-wall cost is separate.',
        results_identity=canonical_hash(files), games=games,
        interpretation='Exploratory calibration, not held-out agent testing. Tiers are relative to one fixed anchor, not a global or transitive ranking. Unmeasured pairings are unknown. Nominal intervals are not multiplicity-corrected.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output/'calibration.json', summary)
    write_json(output/'opponent_catalog.json', dict(format='boardbench_joint_opponent_catalog_v1',
        pool_sha256=manifest['manifest_sha256'], calibration_identity=summary['results_identity'],
        games={g:dict(roles=d['roles'], members={n:dict(policy_id=manifest['games'][g]['members'][n]['policy_id'],
            strength=label, budget_violations=d['budget_violations'][n]) for n, label in d['strength'].items()}) for g, d in games.items()},
        note='Load policies by the immutable pool/game/member name. Role labels are exploratory, not a total order.'))
    lines = ['# V1.7.4 联合对手池标定', '',
        f"已记录 {summary['recorded']}/{summary['scheduled']} 局；状态：`{dict(statuses)}`。", '',
        '分档以各游戏冻结 legacy 为锚点。高／低要求完整面板、至少 16 个独立环境，胜率至少 60%／至多 40%，且名义区间在 50% 同侧。中间未决不代表等强。', '',
        '区间对环境内两座位聚合后 bootstrap，并与按环境数计算的 Wilson 区间取包络，避免小样本全胜产生零宽区间；是探索性区间，未作多重比较校正。', '',
        '| 游戏 | 锚点以下 | 锚点以上 | 未拉开差距／证据不足 |', '|---|---|---|---|']
    for game, data in games.items():
        names = lambda value: ', '.join(n for n, v in data['strength'].items() if v == value) or '—'
        uncertain = ', '.join(n+(' [超预算]' if v == 'budget_violation' else '') for n, v in data['strength'].items() if v in ('unresolved_relative_to_anchor', 'insufficient_evidence', 'budget_violation')) or '—'
        lines.append(f"| {game} | {names('below_anchor')} | {names('above_anchor')} | {uncertain} |")
    lines += ['', '## 对共同锚点的胜率', '', '| 游戏 | 对手 | 完成/计划 | 胜率 | 名义区间 | 决策秒/局 |', '|---|---|---:|---:|---|---:|']
    for game, data in games.items():
        for name, stat in data['relative_to_anchor'].items():
            rate = '—' if stat['win_rate_complete'] is None else f"{100*stat['win_rate_complete']:.1f}%"
            ci = '—' if stat['interval'] is None else '–'.join(f'{100*x:.1f}%' for x in stat['interval'])
            seconds = stat['mean_policy_seconds'][0]
            lines.append(f"| {game} | {name} | {stat['complete']}/{stat['scheduled']} | {rate} | {ci} | {seconds:.3f} |" if seconds is not None else f'| {game} | {name} | 0/{stat["scheduled"]} | — | — | — |')
    lines += ['', '## 风格证据', '', '以下仅描述相同环境、相同 legacy 对手下的分项差异，相对于 legacy 自对弈。不是因果性的偏好证明；完整动作类型比例及各分项均值见 calibration.json。', '']
    for game, data in games.items():
        for name, profile in data['styles'].items():
            shifts = profile['descriptive_component_shifts']
            if shifts:
                lines.append(f"- {game}/{name}："+'；'.join(f"{x['component']} {x['delta']:+.2f}" for x in shifts)+f"（共同环境 {profile['common_anchor_seeds']} 个）。")
    lines += ['', 'Five Tribes 不纳入本池。规则边界、未决局与超时保留原状态；未知结果上下界见 JSON。固定种子不保证计时搜索逐动作相同，需同时记录硬件与资源。']
    (output/'Calibration_zh.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pool', 'schedule', 'results', 'output'): p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args(); result = report(a.pool, a.schedule, a.results, a.output)
    print(json.dumps({k: result[k] for k in ('scheduled', 'recorded', 'statuses', 'recorded_match_cpu_hours')}))


if __name__ == '__main__': main()
