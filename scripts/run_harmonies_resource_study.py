#!/usr/bin/env python3
"""Preregister and run paired historical/budget experiments; no result-driven selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time

from boardbench.families.pilot_run import (
    Budget, Checkpoints, NativePilotAdapter, PilotConfig, PilotRunner, gateway_callbacks,
    prepare_public_material, export_known_catalog, tree_hash, validate_submission, _write_json,
)


LEVELS = (
    ('b025',900,8,25,500000,64000),
    ('b050',1800,16,50,1000000,128000),
    ('b100',3600,32,100,2000000,256000),
    ('b200',7200,64,200,4000000,512000),
)


def bundles():
    return {name:Budget(gt,calls,inp,out,8192,wall,8,8*wall,32)
            for name,wall,gt,calls,inp,out in LEVELS}


def semantic_hash(path):
    h=hashlib.sha256()
    for f in sorted(Path(path).rglob('*')):
        if f.is_file():
            value=f.read_bytes()
            if f.name=='manifest.json':value=json.dumps(json.loads(value),sort_keys=True).encode()
            h.update(str(f.relative_to(path)).encode()+b'\0'+value+b'\0')
    return h.hexdigest()


def historical_versions(source):
    """Order from successful controller outputs, never filesystem mtime or scores."""
    source=Path(source)
    if json.loads((source/'state.json').read_text())['status']!='complete':
        raise ValueError('historical development must be complete')
    seen=[]
    for logfile in sorted((source/'phases/development').glob('ledger.*.stdout')):
        for line in logfile.read_text().splitlines():
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')!='item.completed' or item.get('type')!='command_execution' or item.get('exit_code')!=0:
                continue
            for sha in re.findall(r"['\"]sha256['\"]\s*:\s*['\"]([0-9a-f]{64})",item.get('aggregated_output','')):
                if sha not in seen:seen.append(sha)
    versions=[];semantic={}
    for index,sha in enumerate(seen):
        p=source/'checkpoints'/sha;validate_submission(p)
        if tree_hash(p)!=sha:raise ValueError('historical hash mismatch')
        digest=semantic_hash(p)
        if digest not in semantic:
            semantic[digest]=len(versions)
            versions.append(dict(label=f'history-v{len(versions)+1}',sha256=sha,
                                 first_observed_order=index+1,aliases=[],semantic_sha256=digest))
        versions[semantic[digest]]['aliases'].append(sha)
    if not versions:raise ValueError('no published historical versions found')
    return versions


def worker(spec):
    from boardbench.families import pilot_gt,pilot_gateway
    from boardbench.families.evaluate import archive_source
    from boardbench.families.pilot_supervisor import runtime_inventory
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=False,mode=0o700)
    _write_json(root/'source_identity.json',archive_source(root))
    identity=pilot_gt.initialize(root/'gt',spec['pool'],seed_key=spec['seed_key'])
    gt=pilot_gt.PilotService(root/'gt')
    upstream=json.loads(Path(spec['private_upstream']).read_text())
    if upstream['model'].split('/')[-1]!='gpt-5.6-sol' or upstream.get('reasoning_effort')!='medium':
        raise ValueError('study pins GPT-5.6-sol medium')
    gateway=pilot_gateway.Gateway(root/'gateway',upstream)
    servers=[]
    try:
        with tempfile.TemporaryDirectory(prefix='curve-sockets-') as directory:
            gt_socket=Path(directory)/'gt.sock';model_socket=Path(directory)/'model.sock'
            servers=[pilot_gt.make_server(gt,unix_socket=gt_socket),pilot_gateway.make_server(gateway,unix_socket=model_socket)]
            for server in servers:threading.Thread(target=server.serve_forever,daemon=True).start()
            sdk=root/'public'
            prepare_public_material(sdk,Path(pilot_gt.__file__).with_name('pilot_client.py'),[Path(spec['rules'])])
            export_known_catalog(sdk/'rules')
            account,status=gateway_callbacks(gateway)
            adapter=NativePilotAdapter(gt_service=gt,gt_socket=gt_socket,gateway_socket=model_socket,
                sdk_dir=sdk,gateway_account=account,gateway_status=status,gateway_close=gateway.close_model,
                model_id='gpt-5.6-sol')
            _write_json(root/'runtime_inventory.json',runtime_inventory(adapter.runtime))
            config=PilotConfig(opponent_id=identity['policy_id'],opponent_sha256=identity['weights_sha256'],
                environment_revision=identity['environment_source_sha256'],codex_version='0.167.0',
                deadline_unix=spec['deadline'],model_id='gpt-5.6-sol',protocol_version='resource_curve_v1_20261116',
                calibration_pairs=16,include_controls=False,deployment_profiles=('L20',),
                calibration_seeds=tuple(pilot_gt.private_seed(spec['seed_key'],'calibration',i) for i in range(16)),
                development_only=spec['kind']=='development',
                development_budget=Budget(**spec['budget']) if 'budget' in spec else bundles()['b100'])
            runner=PilotRunner(root/'experiment',config,adapter)
            if spec['kind']=='development':result=runner.run()
            else:result=runner.run_archived_snapshot(Path(spec['source_experiment']),spec['sha256'])
            print(json.dumps(dict(event='worker_end',label=spec['label'],status=result['status'])),flush=True)
            if spec['kind']=='evaluation':
                from report_harmonies_pilot import summarize
                _,summary=summarize(root/'experiment');_write_json(root/'experiment/summary.json',summary)
    finally:
        for server in servers:server.shutdown();server.server_close()
        gt.close();gateway.close()


def report(root,plan):
    rows=[]
    for label in [v['label'] for v in plan['historical_versions']]+[d['label'] for d in plan['developments']]:
        evaluation=root/('eval-'+label)/'experiment'
        record=dict(label=label,status='pending',recorded=0)
        path=evaluation/'state.json'
        if path.exists():record['status']=json.loads(path.read_text())['status']
        path=evaluation/'summary.json'
        if path.exists():
            group=json.loads(path.read_text())['groups'].get('developed-L20',{})
            record.update(group)
        dev=root/('dev-'+label)/'experiment'
        if (dev/'state.json').exists():
            state=json.loads((dev/'state.json').read_text());record['development_status']=state['status']
            if state['status']=='no_valid_submission':record['status']='no_valid_submission'
            elif state['status'] in ('infrastructure_failure','deadline'):record['status']='development_'+state['status']
            ledger=dev/'phases/development/ledger.json'
            if ledger.exists():
                compute=json.loads(ledger.read_text());record['development_compute']={k:compute.get(k) for k in ('cpu_seconds','active_wall_seconds','running')}
            import sqlite3
            with sqlite3.connect(root/('dev-'+label)/'gateway/gateway.sqlite') as db:
                raw=db.execute("SELECT data FROM accounts WHERE id='development'").fetchone()
                if raw:record['development_model']=json.loads(raw[0])['usage']
        for prefix in ('dev-','eval-'):
            marker=root/(prefix+label+'.exit.json')
            if marker.exists():
                terminal=json.loads(marker.read_text())
                if terminal['exit_code']!=0:record['status']=prefix+'worker_failure'
        rows.append(record)
    _write_json(root/'study_summary.json',dict(rows=rows,interpretation='Calibration only; paired seed/seat, three independent developments per budget; no hidden-test selection. Scores use natural games; failures and missing submissions remain separate.'))
    return rows


def study(args):
    from boardbench.families.pilot_supervisor import physical_cpu_ids
    from boardbench.families.evaluate import archive_source
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    private=root/'private';private.mkdir(mode=0o700)
    versions=historical_versions(args.source_experiment.resolve())
    started=time.time();deadline=started+12*3600
    cores=physical_cpu_ids(24)
    # Own allocation is 24 physical cores, always below half this host.
    try:physical_cpu_ids(48)
    except ValueError:raise RuntimeError('study allocation would exceed half host')
    seed_keys={f'dev-r{r}':secrets.token_hex(32) for r in range(3)}
    seed_keys['evaluation']=secrets.token_hex(32)
    _write_json(private/'seed_keys.json',seed_keys);(private/'seed_keys.json').chmod(0o600)
    devs=[dict(label=f'{level}-r{repeat}',level=level,repeat=repeat,budget=asdict(budget))
          for level,budget in bundles().items() for repeat in range(3)]
    plan=dict(protocol='resource_curve_v1_20261116',model='gpt-5.6-sol',reasoning_effort='medium',
        started=started,deadline=deadline,historical_source=str(args.source_experiment.resolve()),historical_versions=versions,
        developments=devs,evaluation_pairs=16,deployment='L20_C120_G0',opponent='legacy',
        allocation=dict(developer_slots=[list(cores[:8]),list(cores[8:16])],evaluation_slot=list(cores[16:24])),
        seed_commitments={key:hashlib.sha256(value.encode()).hexdigest() for key,value in seed_keys.items()},
        selection='last successful submission; no hidden feedback or best-of-test selection',
        historical_x_axis='first publication order; exact historical resource samples unavailable')
    _write_json(root/'preregistered_plan.json',plan)
    _write_json(root/'source_identity.json',archive_source(root))
    evaluation_queue=queue.Queue();development_queue=queue.Queue();finished=threading.Event();lock=threading.RLock()
    base=dict(pool=str(args.pool.resolve()),rules=str(args.rules.resolve()),private_upstream=str(args.private_upstream.resolve()),deadline=deadline)
    for v in versions:
        evaluation_queue.put(dict(kind='evaluation',label=v['label'],source_experiment=str(args.source_experiment.resolve()),sha256=v['sha256']))
    for d in devs:development_queue.put(d)

    def launch(spec,assigned):
        if time.time()>=deadline:return False
        name=('dev-' if spec['kind']=='development' else 'eval-')+spec['label']
        specification=dict(base,**spec,root=str(root/name))
        path=private/(name+'.json');_write_json(path,specification);path.chmod(0o600)
        with (root/(name+'.log')).open('w') as log:
            process=subprocess.Popen(['/usr/bin/taskset','-c',','.join(map(str,assigned)),sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(path)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=process.wait(timeout=max(1,deadline-time.time()+5))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        _write_json(root/(name+'.exit.json'),dict(exit_code=code,wall_deadline=timed_out,ended=time.time()))
        with lock:
            report(root,plan)
            print(json.dumps(dict(event='task_end',task=name,exit_code=code)),flush=True)
        return code==0

    def developer(slot):
        while time.time()<deadline:
            try:d=development_queue.get_nowait()
            except queue.Empty:return
            spec=dict(kind='development',label=d['label'],budget=d['budget'],seed_key=seed_keys[f"dev-r{d['repeat']}"])
            ok=launch(spec,cores[slot*8:slot*8+8])
            if ok:
                experiment=root/('dev-'+d['label'])/'experiment'
                state=json.loads((experiment/'state.json').read_text())
                if state['status']=='complete' and state.get('frozen_submission'):
                    sha=Path(state['frozen_submission']).name
                    evaluation_queue.put(dict(kind='evaluation',label=d['label'],source_experiment=str(experiment),sha256=sha))
            development_queue.task_done()

    def evaluator():
        while not finished.is_set() or not evaluation_queue.empty():
            if time.time()>=deadline:return
            try:spec=evaluation_queue.get(timeout=1)
            except queue.Empty:continue
            spec['seed_key']=seed_keys['evaluation']
            launch(spec,cores[16:24]);evaluation_queue.task_done()

    print(json.dumps(dict(event='study_start',historical=len(versions),developments=len(devs),deadline=deadline)),flush=True)
    with ThreadPoolExecutor(3) as executor:
        evaluation=executor.submit(evaluator)
        developers=[executor.submit(developer,i) for i in range(2)]
        try:
            for future in developers:future.result()
        finally:finished.set()
        evaluation.result()
    rows=report(root,plan)
    _write_json(root/'finished.json',dict(ended=time.time(),deadline_reached=time.time()>=deadline,
                                       completed_evaluations=sum(row['status']=='complete' for row in rows)))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for row in rows:
            if row.get('own_score_mean') is None:continue
            label=row['label'];score=row['own_score_mean']
            if label.startswith('history'):
                axes[0].scatter(int(label.split('v')[1]),score,color='tab:blue')
            else:
                axes[1].scatter(int(label[1:4])/100,score,color='tab:orange')
        axes[0].set(xlabel='Historical strategy version',ylabel='Mean score (natural completions)')
        axes[1].set(xlabel='Development resource multiplier',ylabel='Mean score per independent run')
        for ax in axes:ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(root/'score_curves.png',dpi=160);plt.close(fig)
    except ImportError:pass
    from audit_pilot_resources import audit
    _write_json(root/'resource_audit.json',audit(root))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path)
    parser.add_argument('--root',type=Path);parser.add_argument('--source-experiment',type=Path)
    parser.add_argument('--pool',type=Path);parser.add_argument('--private-upstream',type=Path);parser.add_argument('--rules',type=Path)
    args=parser.parse_args()
    if args.worker:worker(json.loads(args.worker.read_text()))
    else:
        if not all((args.root,args.source_experiment,args.pool,args.private_upstream,args.rules)):
            parser.error('study requires root, source experiment, pool, private upstream and rules')
        study(args)


if __name__=='__main__':main()
