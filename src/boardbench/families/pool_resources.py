"""Bounded, read-only node-local telemetry for running common-policy pilots.

No model imports, CUDA initialization, job signaling or allocation changes.
Private process/node details stay in ignored output directories. Shared-device
utilization is reported per physical device, never attributed to every job.
"""
import argparse
from collections import defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def parse_proc_stat(text, ticks, page_bytes):
    # comm can contain spaces and parentheses; fields after its last ')' are fixed.
    fields = text.rsplit(') ', 1)[1].split()
    return dict(ppid=int(fields[1]), start_ticks=int(fields[19]),
                cpu_seconds=(int(fields[11])+int(fields[12])+int(fields[13])+int(fields[14]))/ticks,
                rss_bytes=int(fields[21])*page_bytes)


def process_table():
    ticks, page_bytes = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGE_SIZE')
    table = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            table[int(proc.name)] = parse_proc_stat((proc/'stat').read_text(), ticks, page_bytes)
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(root, table):
    children = {root}
    while True:
        expanded = children | {pid for pid, row in table.items() if row['ppid'] in children}
        if expanded == children:
            return children
        children = expanded


def gpu_snapshot(indices):
    def query(fields, kind):
        cmd = ['nvidia-smi', '-i', ','.join(map(str, indices)), '--query-'+kind+'='+fields,
               '--format=csv,noheader,nounits']
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return list(csv.reader(io.StringIO(p.stdout))) if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    cards = query('index,uuid,name,memory.total,memory.used,utilization.gpu', 'gpu')
    apps = query('gpu_uuid,pid,used_gpu_memory', 'compute-apps')
    devices, processes = [], []
    for row in cards or []:
        try:
            index, uuid, name, total, used, util = [x.strip() for x in row]
            devices.append(dict(index=int(index), uuid=uuid, name=name,
                                total_mib=int(total), used_mib=int(used), util_percent=int(util)))
        except (ValueError, TypeError):
            continue
    for row in apps or []:
        try:
            uuid, pid, used = [x.strip() for x in row]
            processes.append(dict(uuid=uuid, pid=int(pid), used_mib=int(used)))
        except (ValueError, TypeError):
            continue
    return dict(devices=devices, processes=processes, query_ok=cards is not None and apps is not None)


def cpu_rate(current, previous, now, then):
    """CPU-seconds / elapsed seconds: one fully busy logical CPU equals 1.0."""
    if (previous is None or not current.get('alive') or not previous.get('alive')
            or current['process_identity'] != previous['process_identity'] or now <= then):
        return None
    delta = current['cpu_seconds']-previous['cpu_seconds']
    return delta/(now-then) if delta >= 0 else None


def learner_config(args):
    # A timeout wrapper contains the same module/config strings. Only the
    # Python learner is a root, never its wrapper or unrelated descendants.
    if not args or not Path(args[0]).name.startswith('python'):
        return None
    try:
        if args[args.index('-m')+1] != 'boardbench.families.pool_training':return None
        return args[args.index('--config')+1]
    except (ValueError, IndexError):
        return None


def sample(jobs, previous=None, include_pss=False):
    began = time.monotonic()
    table = process_table()
    roots = {}
    expected = {str(Path(j['output'])/'config.json'): j for j in jobs}
    for pid in table:
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split(chr(0))
            config = learner_config(args)
            if config in expected:
                roots[config] = pid
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    gpu = gpu_snapshot(sorted({j['gpu'] for j in jobs}))
    result = dict(unix=time.time(), monotonic=time.monotonic(), jobs=[], gpu=gpu)
    prior = {j['job']:j for j in previous['jobs']} if previous else {}
    for j in jobs:
        output = Path(j['output'])
        state = read_json(output/'state.json', {})
        row = dict(job=j['game']+'/'+j['arm'], game=j['game'], arm=j['arm'],
            allocated_cpus=len(j['cpu_ids']), gpu=j['gpu'], phase=state.get('phase'),
            learner_decisions=state.get('learner_decisions',0), next_update=state.get('next_update',0),
            costs=state.get('costs',{}), output=str(output),
            has_exit_file=(output/'exit_code').exists(), deadline=j['deadline_unix'])
        pid = roots.get(str(output/'config.json'))
        if pid is None:
            row.update(alive=False, cpu_core_equivalents=None)
        else:
            group = descendants(pid, table)
            allowed, violations, pss = set(j['cpu_ids']), [], []
            for child in group:
                try:
                    if not set(os.sched_getaffinity(child)) <= allowed:
                        violations.append(child)
                    if include_pss:
                        lines = Path(f'/proc/{child}/smaps_rollup').read_text().splitlines()
                        pss.append(next(int(line.split()[1])*1024 for line in lines if line.startswith('Pss:')))
                except (OSError, StopIteration):
                    pass
            row.update(alive=True, process_identity=[pid,table[pid]['start_ticks']], processes=len(group),
                cpu_seconds=sum(table[p]['cpu_seconds'] for p in group),
                rss_sum_mib=sum(table[p]['rss_bytes'] for p in group)/2**20,
                pss_sum_mib=sum(pss)/2**20 if len(pss)==len(group) else None,
                affinity_violations=violations,
                gpu_memory_mib=sum(p['used_mib'] for p in gpu['processes'] if p['pid'] in group) if gpu['query_ok'] else None)
            row['cpu_core_equivalents'] = cpu_rate(row, prior.get(row['job']),
                result['monotonic'], previous['monotonic'] if previous else result['monotonic'])
            row['cpu_sample_seconds'] = result['monotonic']-previous['monotonic'] if previous else 0.
        result['jobs'].append(row)
    result['monitor_seconds'] = time.monotonic()-began
    return result


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered)-1)*fraction
    lower = math.floor(point)
    return ordered[lower] + (ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(point-lower)


def summarize(samples):
    grouped, cards = defaultdict(list), defaultdict(list)
    for s in samples:
        for row in s['jobs']:
            grouped[row['job']].append(row)
        for card in s['gpu']['devices']:
            cards[card['uuid']].append(card)
    jobs = []
    for key, rows in sorted(grouped.items()):
        rates = [r for r in rows if r.get('cpu_core_equivalents') is not None]
        seconds = sum(r['cpu_sample_seconds'] for r in rates)
        busy = sum(r['cpu_sample_seconds']*r['cpu_core_equivalents'] for r in rates)
        live = [r for r in rows if r['alive']]
        def maximum(field):
            return max((r[field] for r in live if r.get(field) is not None),default=None)
        last = rows[-1]
        jobs.append(dict(job=key, game=last['game'], arm=last['arm'], allocated_cpus=last['allocated_cpus'],
            sampled_intervals=len(rates), observed_seconds=seconds,
            cpu_busy_seconds=busy, cpu_mean=busy/seconds if seconds else None,
            cpu_p95=quantile([r['cpu_core_equivalents'] for r in rates],.95),
            cpu_max=maximum('cpu_core_equivalents'), rss_sum_peak_mib=maximum('rss_sum_mib'),
            pss_peak_mib=maximum('pss_sum_mib'), gpu_memory_peak_mib=maximum('gpu_memory_mib'),
            affinity_violations=sum(len(r.get('affinity_violations',[])) for r in rows),
            phase=last['phase'], alive=last['alive'], has_exit_file=last['has_exit_file']))
    # Intentionally anonymous physical-device rows: never duplicate shared
    # utilization across jobs or infer per-job GPU-seconds from this metric.
    devices = [dict(device_number=i, samples=len(rows), name=rows[-1]['name'],
                    gpu_util_mean=statistics.mean(r['util_percent'] for r in rows),
                    gpu_util_p95=quantile([r['util_percent'] for r in rows],.95),
                    memory_peak_mib=max(r['used_mib'] for r in rows))
               for i, rows in enumerate(cards.values())]
    return dict(samples=len(samples), first_unix=samples[0]['unix'] if samples else None,
                last_unix=samples[-1]['unix'] if samples else None, jobs=jobs, devices=devices,
                monitor_seconds=sum(s.get('monitor_seconds',0.) for s in samples),
                limitations=['Monitoring begins after training launch; peaks are observed, not lifetime peaks.',
                    'RSS sums double-count shared pages; use PSS when available.',
                    'GPU utilization belongs to the shared physical card, not each learner.',
                    'No CPU/GPU allocation sweep was performed; usage is not a tested minimum requirement.'])


def read_samples(path):
    rows=[]
    try:
        with Path(path).open() as handle:
            for line in handle:
                try:rows.append(json.loads(line))
                except ValueError:continue
    except OSError:pass
    return rows


def combined_report(directory):
    nodes = [summarize(read_samples(p)) for p in sorted(Path(directory).glob('*.jsonl'))]
    jobs=[]
    for node in nodes:
        for job in node['jobs']:
            # Only state/config files, never trajectories or model tensors.
            jobs.append(job)
    return dict(observed_unix=time.time(), nodes=nodes, jobs=jobs)


def enrich_report(summary, jobs):
    by_id={j['game']+'/'+j['arm']:j for j in jobs}
    for row in summary['jobs']:
        output=Path(by_id[row['job']]['output'])
        state=read_json(output/'state.json',{})
        config=read_json(output/'config.json',{})
        origin=config.get('resume_provenance',{})
        end=(output/'exit_code').stat().st_mtime if (output/'exit_code').exists() else time.time()
        elapsed=max(0.,end-state.get('began_unix',end))
        costs={k:v-origin.get('starting_costs',{}).get(k,0.) for k,v in state.get('costs',{}).items()}
        decisions=state.get('learner_decisions',0)-origin.get('starting_learner_decisions',0)
        row.update(learner_decisions=state.get('learner_decisions',0),stage_learner_decisions=decisions,wall_seconds=elapsed,
            decisions_per_wall_second=decisions/elapsed if elapsed else None,
            cost_seconds=costs, cost_wall_fractions={k:v/elapsed for k,v in costs.items()} if elapsed else {},
            monitored_generation_only=True)
    return summary


def markdown_report(summary):
    lines=['# Pilot resource observations', '',
           'Observed peaks and sampled CPU usage; not a minimum-resource guarantee.',
           'Shared GPU utilization is reported separately from per-job GPU memory.', '',
           '| Game | Arm | Decisions/s (wall) | CPU mean / p95 | PSS peak GiB | GPU memory peak GiB | Update wall % | Save wall % |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return '—' if value is None else f'{value:.2f}'
    for j in sorted(summary['jobs'],key=lambda j:j['job']):
        fractions=j.get('cost_wall_fractions',{})
        lines.append(f"| {j['game']} | {j['arm']} | {fmt(j.get('decisions_per_wall_second'))} | "
                     f"{fmt(j['cpu_mean'])} / {fmt(j['cpu_p95'])} | "
                     f"{fmt(j['pss_peak_mib']/1024 if j['pss_peak_mib'] is not None else None)} | "
                     f"{fmt(j['gpu_memory_peak_mib']/1024 if j['gpu_memory_peak_mib'] is not None else None)} | "
                     f"{fmt(100*fractions.get('update_seconds',0))} | {fmt(100*fractions.get('io_seconds',0))} |")
    return '\n'.join(lines)+'\n'


def watch(args):
    jobs=read_json(args.jobs)
    if not jobs or args.sample_seconds<5 or args.summary_seconds<args.sample_seconds:
        raise ValueError('Expected explicit jobs and a low-overhead sampling interval')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'samples.jsonl'
    previous=None
    samples=read_samples(raw)
    deadline=max(j['deadline_unix'] for j in jobs)+60
    next_summary=0.
    while True:
        before=time.monotonic()
        emit=time.time()>=next_summary
        current=sample(jobs,previous,include_pss=emit)
        samples.append(current)
        with raw.open('a') as handle:handle.write(json.dumps(current,allow_nan=False)+'\n')
        final=(all(r['has_exit_file'] for r in current['jobs']) or time.time()>=deadline)
        if emit or final:
            summary=enrich_report(summarize(samples),jobs)
            summary['final']=final
            write_json(args.output/'summary.json',summary)
            (args.output/'summary.md').write_text(markdown_report(summary))
            print(json.dumps(dict(unix=time.time(),samples=len(samples),live=sum(j['alive'] for j in current['jobs']),
                                  final=final,monitor_seconds=current['monitor_seconds'])),flush=True)
            next_summary=time.time()+args.summary_seconds
        if final:break
        previous=current
        time.sleep(max(.1,args.sample_seconds-(time.monotonic()-before)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-seconds',type=float,default=15.)
    p.add_argument('--summary-seconds',type=float,default=300.)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.once:
        jobs=read_json(args.jobs);s=sample(jobs,include_pss=True)
        write_json(args.output,enrich_report(summarize([s]),jobs))
    else:watch(args)


if __name__=='__main__':main()
