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
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

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
    """Claim one exact trial; parallel dispatch may await it, never replay it."""
    import fcntl
    root=Path(spec['root'])
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    completion=root/'worker_completion.json'
    with (root.parent/(root.name+'.worker.lock')).open('a') as lock:
        while True:
            if time.time()>=spec['deadline']:raise RuntimeError('worker claim deadline reached')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:time.sleep(min(.2,max(0,spec['deadline']-time.time())))
        if root.exists():
            if not completion.exists():raise RuntimeError('previous worker lacks completion record; no automatic replay')
            saved=json.loads(completion.read_text())
            if saved['spec_sha256']!=fingerprint:raise ValueError('existing worker specification differs')
            if saved['status'] not in ('complete','no_valid_submission'):
                raise RuntimeError('previous worker failed; no automatic replay')
            print(json.dumps(dict(event='worker_result_reused',label=spec['label'],status=saved['status'])),flush=True)
            return
        metadata=dict(spec_sha256=fingerprint,pid=os.getpid(),cpu_ids=sorted(os.sched_getaffinity(0)),started=time.time())
        try:
            _run_worker(spec)
        except BaseException as error:
            if root.exists():
                _write_json(completion,dict(metadata,status='failed',ended=time.time(),error_type=type(error).__name__))
            raise
        state=json.loads((root/'experiment/state.json').read_text())
        _write_json(completion,dict(metadata,status=state['status'],ended=time.time()))


def _run_worker(spec):
    from boardbench.families import pilot_gt,pilot_gateway
    from boardbench.families.evaluate import archive_source
    from boardbench.families.pilot_supervisor import runtime_inventory
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=False,mode=0o700)
    _write_json(root/'source_identity.json',archive_source(root))
    identity=pilot_gt.initialize(root/'gt',spec['pool'],seed_key=spec['seed_key'])
    gt=pilot_gt.PilotService(root/'gt')
    upstream=json.loads(Path(spec['private_upstream']).read_text())
    model=spec.get('model_id','gpt-5.6-sol')
    if model not in ('gpt-5.6-sol','gpt-6-astra') or upstream['model'].split('/')[-1]!=model or upstream.get('reasoning_effort')!='medium':
        raise ValueError('private route must match the declared study model and medium effort')
    if spec.get('api_interval_seconds',0):
        upstream.update(transport_interval_seconds=spec['api_interval_seconds'],transport_clock_path=spec['api_clock'])
    if spec.get('wait_for_network'):
        upstream['count_retry_delays_seconds']=[30,60]
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
                model_id=model)
            _write_json(root/'runtime_inventory.json',runtime_inventory(adapter.runtime))
            pairs=spec.get('evaluation_pairs',16)
            config=PilotConfig(opponent_id=identity['policy_id'],opponent_sha256=identity['weights_sha256'],
                environment_revision=identity['environment_source_sha256'],codex_version='0.167.0',
                deadline_unix=spec['deadline'],model_id=model,protocol_version=spec.get('protocol','resource_curve_v1_20261116'),
                calibration_pairs=pairs,include_controls=False,deployment_profiles=('L20',),
                calibration_seeds=tuple(pilot_gt.private_seed(spec['seed_key'],'calibration',i) for i in range(pairs)),
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
    from report_harmonies_pilot import summarize
    for retained in plan.get('retained_evaluations',[]):
        source=Path(retained['source_experiment'])
        if hashlib.sha256((source/'results.jsonl').read_bytes()).hexdigest()!=retained['results_sha256']:
            raise ValueError('retained evaluation changed')
        _,summary=summarize(source,pair_limit=plan['evaluation_pairs'])
        rows.append(dict(label=retained['label'],status='complete',reused=True,
                         source_experiment=str(source),**summary['groups']['developed-L20']))
    for label in ([v['label'] for v in plan['historical_versions']]
                  +[v['label'] for v in plan.get('additional_evaluations',[])]
                  +[d['label'] for d in plan['developments']]):
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
    _write_json(root/'study_summary.json',dict(rows=rows,interpretation=(
        f"Exploratory calibration: {plan.get('development_repeats',3)} development(s) per budget, "
        f"{plan['evaluation_pairs']} paired seed(s); no hidden-test selection. "
        'Scores use natural games; failures, missing submissions and reused evaluations remain separate.')))
    return rows


def paired_seed_keys(root):
    root=Path(root)
    plan=json.loads((root/'preregistered_plan.json').read_text())
    keys=json.loads((root/'private/seed_keys.json').read_text())
    commitments={key:hashlib.sha256(value.encode()).hexdigest() for key,value in keys.items()}
    if commitments!=plan['seed_commitments'] or set(keys)!={'dev-r0','dev-r1','dev-r2','evaluation'}:
        raise ValueError('paired seed commitments do not match')
    expected=[asdict(bundles()[d['level']])==d['budget'] for d in plan['developments']]
    if len(expected)!=12 or not all(expected):raise ValueError('paired development budgets differ')
    return keys


def infrastructure_failed(root,exit_code):
    if exit_code!=0:return True
    path=Path(root)/'experiment/state.json'
    return path.exists() and json.loads(path.read_text()).get('status')=='infrastructure_failure'


def known_preflight_network_failure(root):
    """Only recognize an explicit native error, not an earlier recovered diagnostic."""
    for path in (Path(root)/'experiment/phases').glob('*/ledger.*.stdout'):
        for line in path.read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('type') not in ('error','turn.failed'):continue
            message=event.get('message','') or (event.get('error') or {}).get('message','')
            if any(code in message for code in ('upstream_dns_failure','input_count_transport_failure')):
                return True
    return False


def wait_for_network(private_upstream,deadline,event_path,*,not_before=0,clock=time.time,sleep=time.sleep,resolve=socket.getaddrinfo):
    """Low-frequency DNS only; never probe model generations or reset a budget."""
    endpoint=urlsplit(json.loads(Path(private_upstream).read_text())['base_url'])
    def event(kind,**data):
        with Path(event_path).open('a') as stream:
            stream.write(json.dumps(dict(event=kind,time=clock(),**data))+'\n')
    def wait(seconds):
        end=min(deadline,clock()+seconds)
        event('network_wait',seconds=max(0,end-clock()))
        while clock()<end:sleep(min(1.,end-clock()))
    if not_before>clock():wait(not_before-clock())
    attempt=0
    while clock()<deadline:
        attempt+=1
        try:
            addresses=resolve(endpoint.hostname,endpoint.port or (443 if endpoint.scheme=='https' else 80),type=socket.SOCK_STREAM)
            if not addresses:raise OSError('empty resolver response')
        except OSError as error:
            event('dns_unavailable',attempt=attempt,error_type=type(error).__name__,errno=error.errno)
            wait((60,120,300)[min(attempt-1,2)])
            continue
        event('dns_ready',attempt=attempt)
        return clock()<deadline
    event('network_wait_deadline')
    return False


def continuation_selection(parent,developments,pairs,model):
    """Keep completed evidence; only dispatch trials which never started."""
    parent=Path(parent)
    plan=json.loads((parent/'preregistered_plan.json').read_text())
    if not (parent/'finished.json').exists():raise ValueError('parent study is not stopped')
    if plan['model']!=model or pairs>plan['evaluation_pairs']:raise ValueError('continuation model/pairs mismatch')
    remaining=[];retained=[];evaluations=[];omitted=[]
    for spec in developments:
        label=spec['label'];dev=parent/('dev-'+label);state=dev/'experiment/state.json'
        if not dev.exists() and not (parent/'private'/('dev-'+label+'.json')).exists():
            remaining.append(spec);continue
        if not state.exists():
            omitted.append(dict(label=label,reason='previously_started_without_state'));continue
        previous=json.loads(state.read_text())
        if previous['status']!='complete' or not previous.get('frozen_submission'):
            omitted.append(dict(label=label,reason='previous_'+previous['status']));continue
        evaluation=parent/('eval-'+label)/'experiment'
        evaluation_state=evaluation/'state.json'
        if evaluation_state.exists() and json.loads(evaluation_state.read_text())['status']=='complete':
            retained.append(dict(label=label,source_experiment=str(evaluation.resolve()),
                                 results_sha256=hashlib.sha256((evaluation/'results.jsonl').read_bytes()).hexdigest()))
        elif not evaluation.exists():
            sha=Path(previous['frozen_submission']).name
            evaluations.append(dict(kind='evaluation',label=label,source_experiment=str((dev/'experiment').resolve()),sha256=sha))
        else:
            omitted.append(dict(label=label,reason='previous_evaluation_started'))
    return plan,remaining,retained,evaluations,omitted


def study(args):
    from boardbench.families.pilot_supervisor import physical_cpu_ids
    from boardbench.families.evaluate import archive_source
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    private=root/'private';private.mkdir(mode=0o700)
    versions=[] if args.skip_history else historical_versions(args.source_experiment.resolve())
    started=time.time();deadline=started+12*3600
    core_count=8*(args.developer_slots+1)
    cores=physical_cpu_ids(core_count)
    # Count physical cores; disjoint developer and evaluator allocations.
    try:physical_cpu_ids(core_count*2)
    except ValueError:raise RuntimeError('study allocation would exceed half host')
    seed_keys={f'dev-r{r}':secrets.token_hex(32) for r in range(3)}
    seed_keys['evaluation']=secrets.token_hex(32)
    paired_source=args.continue_study or args.paired_study
    if paired_source:seed_keys=paired_seed_keys(paired_source)
    _write_json(private/'seed_keys.json',seed_keys);(private/'seed_keys.json').chmod(0o600)
    devs=[dict(label=f'{level}-r{repeat}',level=level,repeat=repeat,budget=asdict(budget))
          for level,budget in bundles().items() for repeat in range(args.development_repeats)]
    retained=[];additional_evaluations=[];omitted=[]
    if args.continue_study:
        parent,devs,retained,additional_evaluations,omitted=continuation_selection(
            args.continue_study,devs,args.evaluation_pairs,args.model_id)
        deadline=parent['deadline']
        if deadline<=time.time():raise ValueError('original study deadline has elapsed')
    plan=dict(protocol='resource_curve_v3_20261116',model=args.model_id,reasoning_effort='medium',
        paired_study=str(paired_source.resolve()) if paired_source else None,
        continuation_of=str(args.continue_study.resolve()) if args.continue_study else None,
        development_repeats=args.development_repeats,retained_evaluations=retained,additional_evaluations=additional_evaluations,omitted_previous_trials=omitted,
        wait_for_network=args.wait_for_network,count_retry_delays_seconds=[30,60] if args.wait_for_network else [.5,1.],
        network_policy='DNS wait 60/120/300 seconds; after exhausted preflight cool down 180 seconds; keep failed trial, continue only unstarted tasks' if args.wait_for_network else 'fail fast',
        api_interval_seconds=args.api_interval_seconds,stop_on_infrastructure_failure=args.stop_on_infrastructure_failure,
        started=started,deadline=deadline,historical_source=str(args.source_experiment.resolve()) if args.source_experiment else None,historical_versions=versions,
        developments=devs,evaluation_pairs=args.evaluation_pairs,deployment='L20_C120_G0',opponent='legacy',
        allocation=dict(developer_slots=[list(cores[i*8:i*8+8]) for i in range(args.developer_slots)],evaluation_slot=list(cores[-8:])),
        seed_commitments={key:hashlib.sha256(value.encode()).hexdigest() for key,value in seed_keys.items()},
        selection='last successful submission; no hidden feedback or best-of-test selection',
        historical_x_axis='first publication order; exact historical resource samples unavailable')
    _write_json(root/'preregistered_plan.json',plan)
    _write_json(root/'source_identity.json',archive_source(root))
    evaluation_queue=queue.Queue();development_queue=queue.Queue();finished=threading.Event();lock=threading.RLock()
    paused=threading.Event();network_not_before=[0.]
    base=dict(pool=str(args.pool.resolve()),rules=str(args.rules.resolve()),private_upstream=str(args.private_upstream.resolve()),deadline=deadline,
              model_id=args.model_id,protocol=plan['protocol'],api_interval_seconds=args.api_interval_seconds,api_clock=str(private/'api_clock'),
              evaluation_pairs=args.evaluation_pairs,wait_for_network=args.wait_for_network)
    for evaluation in additional_evaluations:evaluation_queue.put(evaluation)
    for v in versions:
        evaluation_queue.put(dict(kind='evaluation',label=v['label'],source_experiment=str(args.source_experiment.resolve()),sha256=v['sha256']))
    for d in devs:development_queue.put(d)

    def launch(spec,assigned):
        if time.time()>=deadline or paused.is_set():return False
        name=('dev-' if spec['kind']=='development' else 'eval-')+spec['label']
        if args.wait_for_network and spec['kind']=='development':
            if not wait_for_network(args.private_upstream,deadline,root/'network_events.jsonl',not_before=network_not_before[0]):
                return False
            if paused.is_set():return False
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
        failed=infrastructure_failed(root/name,code)
        if failed:
            with lock:
                if args.wait_for_network and known_preflight_network_failure(root/name):
                    network_not_before[0]=time.time()+180
                    with (root/'network_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(event='failed_trial_retained',task=name,time=time.time(),not_before=network_not_before[0]))+'\n')
                elif args.stop_on_infrastructure_failure:
                    paused.set()
                    _write_json(root/'paused.json',dict(task=name,reason='infrastructure_failure_no_new_dispatch',time=time.time()))
        with lock:
            report(root,plan)
            print(json.dumps(dict(event='task_end',task=name,exit_code=code)),flush=True)
        return code==0 and not failed

    def developer(slot):
        while time.time()<deadline and not paused.is_set():
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
            if time.time()>=deadline or paused.is_set():return
            try:spec=evaluation_queue.get(timeout=1)
            except queue.Empty:continue
            spec['seed_key']=seed_keys['evaluation']
            launch(spec,cores[-8:]);evaluation_queue.task_done()

    print(json.dumps(dict(event='study_start',historical=len(versions),developments=len(devs),deadline=deadline)),flush=True)
    with ThreadPoolExecutor(args.developer_slots+1) as executor:
        evaluation=executor.submit(evaluator)
        developers=[executor.submit(developer,i) for i in range(args.developer_slots)]
        try:
            for future in developers:future.result()
        finally:finished.set()
        evaluation.result()
    rows=report(root,plan)
    _write_json(root/'finished.json',dict(ended=time.time(),status='paused_infrastructure' if paused.is_set() else 'ended',deadline_reached=time.time()>=deadline,
                                       completed_evaluations=sum(row['status']=='complete' for row in rows),
                                       reused_evaluations=sum(bool(row.get('reused')) for row in rows)))
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
    parser.add_argument('--model-id',choices=('gpt-5.6-sol','gpt-6-astra'),default='gpt-5.6-sol')
    parser.add_argument('--developer-slots',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--paired-study',type=Path)
    parser.add_argument('--skip-history',action='store_true')
    parser.add_argument('--api-interval-seconds',type=float,default=0.)
    parser.add_argument('--stop-on-infrastructure-failure',action='store_true')
    parser.add_argument('--development-repeats',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--evaluation-pairs',type=int,default=16)
    parser.add_argument('--continue-study',type=Path)
    parser.add_argument('--wait-for-network',action='store_true')
    args=parser.parse_args()
    if not 0<=args.api_interval_seconds<=60:parser.error('API interval must be between 0 and 60 seconds')
    if not 1<=args.evaluation_pairs<=1024:parser.error('evaluation pairs must be between 1 and 1024')
    if args.continue_study and (args.paired_study or not args.skip_history):parser.error('continuation requires skip-history and replaces paired-study')
    if args.worker:worker(json.loads(args.worker.read_text()))
    else:
        if not all((args.root,args.pool,args.private_upstream,args.rules)) or (not args.skip_history and args.source_experiment is None):
            parser.error('study requires root, source experiment, pool, private upstream and rules')
        study(args)


if __name__=='__main__':main()
