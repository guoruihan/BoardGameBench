"""Launch checked immutable family jobs via SSH; private host maps stay external.

The JSON plan provides host/game/kind/slot/gpu/output/hours and optional prior.
Evaluation jobs use kind=eval, bots, phase, seed_start and num_seeds.
Every slot reserves eight CPUs (seven workers plus coordinator). Hostnames map
to previously reachable IPs in a private inventory; do not probe failed hosts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import json, os, subprocess, time, shlex
from pathlib import Path
cfg=json.loads(CONFIG)
def usage():
    v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    return sum(v[:8]),v[3]+v[4]
a=usage();time.sleep(1);b=usage()
util=100*(1-(b[1]-a[1])/max(1,b[0]-a[0]))
cpus=sorted(os.sched_getaffinity(0));limit=len(cpus)//2
if cfg['host'].startswith('a'): limit=min(limit,128)
start=cfg['slot']*8
if start<0 or start+8>limit: raise RuntimeError('half CPU quota exceeded')
chosen=cpus[start:start+8]
if util>55: raise RuntimeError('CPU capacity changed; defer launch: '+str(util))
destination=Path(cfg['output'])
out=destination if cfg['kind']!='eval' else destination.with_name(destination.name+'_job')
out.mkdir(parents=True,exist_ok=True)
manifest=out/'launch.json'
if manifest.exists(): raise RuntimeError('existing launch manifest; inspect before restarting')
active=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
for path in (Path(cfg['root'])/'outputs/v160/bots').glob('**/launch.json'):
    old=json.loads(path.read_text())
    if old.get('session') in active and set(old.get('cpu_ids',[])) & set(chosen):
        raise RuntimeError('CPU overlap with live own session: '+old['session'])
if cfg['kind']=='ppo':
    query=subprocess.run(['nvidia-smi','-i',str(cfg['gpu']),
       '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
       capture_output=True,text=True,check=True,timeout=10)
    used,total,gpu_util=map(int,query.stdout.strip().split(','))
    if total-used<16000 or used>5000 or gpu_util>30: raise RuntimeError('GPU headroom changed; defer launch')
    cfg['gpu_preflight']=dict(used_mib=used,total_mib=total,util_percent=gpu_util)
module={'ppo':'boardbench.families.ppo','cem':'boardbench.families.train',
        'eval':'boardbench.families.evaluate'}[cfg['kind']]
command=[str(Path(cfg['root'])/'.venv-train/bin/python'),'-m',module,'--game',cfg['game'],
 '--output',str(destination),'--workers','7','--deadline-unix',str(cfg['deadline'])]
if cfg['kind']=='eval':
    command+=['--bots',cfg['bots'],'--phase',cfg['phase'],'--seed-start',str(cfg['seed_start']),
              '--num-seeds',str(cfg['num_seeds']),'--wall-seconds',str(cfg.get('hours',1)*3600)]
    if cfg.get('validate'): command+=['--validate']
    if cfg.get('record'): command+=['--record']
    if 'max_match_seconds' in cfg: command+=['--max-match-seconds',str(cfg['max_match_seconds'])]
    for path in cfg.get('exclude_manifests',[]):command+=['--exclude-manifest',path]
else:
    command+=['--hours',str(cfg.get('hours',3)),'--source-commit',cfg['commit']]
if cfg['kind']=='ppo':
    command+=['--device','cuda','--games','32','--updates',str(cfg.get('updates',512)),
              '--gate-seeds','12','--confirm-seeds','64','--seed',str(cfg.get('seed',1601))]
    if cfg.get('prior'): command+=['--prior',cfg['prior']]
    if cfg.get('warm_start'): command+=['--warm-start',cfg['warm_start']]
    for key in ('gamma','gae_lambda','self_play_rate','learning_rate','entropy_coef','prior_kl_coef'):
        if key in cfg: command+=['--'+key.replace('_','-'),str(cfg[key])]
elif cfg['kind']=='cem':
    command+=['--population','16','--generations','80','--train-seeds','3','--gate-seeds','12','--confirm-seeds','64']
    if cfg.get('decision_guard'): command+=['--decision-guard',cfg['decision_guard']]
remaining=min(int(cfg.get('hours',3)*3600+120),int(cfg['deadline']-time.time()))
if remaining<300: raise RuntimeError('insufficient time before hard deadline')
shell='set -uo pipefail\ncd '+shlex.quote(cfg['root'])+'\n'
shell+='export PYTHONPATH='+shlex.quote(cfg['release']+'/src')+'\n'
shell+='export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
shell+='export CUDA_VISIBLE_DEVICES='+shlex.quote(str(cfg['gpu']) if cfg['kind']=='ppo' else '')+'\n'
shell+='taskset -c '+','.join(map(str,chosen))+' timeout --signal=TERM --kill-after=60s '+str(remaining)+'s '
shell+=shlex.join(command)+' 2>&1 | tee '+shlex.quote(str(out/'train.log'))+'\n'
shell+='code=${PIPESTATUS[0]}\nprintf "%s\\n" "$code" > '+shlex.quote(str(out/'exit_code'))+'\nexit "$code"\n'
script=out/'run.sh';script.write_text(shell)
cfg.update(cpu_ids=chosen,cpu_quota=limit,observed_cpu_percent=util,launched_unix=time.time(),
           session='v160_'+cfg['game']+'_'+out.name,command=command)
subprocess.run(['tmux','new-session','-d','-s',cfg['session'],'bash',str(script)],check=True)
temporary=manifest.with_suffix('.part');temporary.write_text(json.dumps(cfg,indent=2))
temporary.chmod(0o600);temporary.replace(manifest)
print(json.dumps({k:cfg[k] for k in ['host','game','kind','session','cpu_ids','launched_unix']}))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--user', default='ruihan')
    args=p.parse_args()
    inventory=json.loads(args.inventory.read_text())
    hosts={x['host'].split('.')[0]:x for x in inventory['hosts'] if x['status']=='ok'}
    jobs=json.loads(args.plan.read_text())
    if not (args.release/'src/boardbench/families').is_dir():
        raise ValueError('Missing immutable source release')
    deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
    groups={}
    for job in jobs:
        if job['kind'] not in ('cem','ppo','eval') or type(job['slot']) is not int:
            raise ValueError('Invalid job kind/CPU slot')
        if job['host'] not in hosts:
            raise ValueError('Host was not previously reachable')
        output=(args.root/job['output']).resolve()
        output.relative_to(args.root.resolve()/'outputs/v160/bots')
        job.update(root=str(args.root.resolve()),release=str(args.release.resolve()),
                   commit=args.commit,deadline=deadline,output=str(output))
        groups.setdefault(job['host'],[]).append(job)
    def launch_host(item):
        host, planned=item
        rows=[]
        # Serialize reservations within one host; different hosts can launch in parallel.
        for job in planned:
            code=REMOTE.replace('CONFIG',repr(json.dumps(job)))
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
                 '-o','GlobalKnownHostsFile=/dev/null','-o','UserKnownHostsFile=/tmp/codex_known_hosts_cluster',
                 '-o','StrictHostKeyChecking=accept-new','-o','CheckHostIP=no',
                 args.user+'@'+hosts[host]['ip'],'python3 -c '+shlex.quote(code)]
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
                row=dict(host=host,game=job['game'],kind=job['kind'],returncode=result.returncode,
                         output=result.stdout,error=result.stderr[-800:])
            except subprocess.TimeoutExpired:
                row=dict(host=host,game=job['game'],kind=job['kind'],returncode=None,
                         error='SSH timeout; inspect manifest before retrying')
            rows.append(row)
            print(json.dumps(row),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=[row for group in pool.map(launch_host,groups.items()) for row in group]
    args.record.parent.mkdir(parents=True,exist_ok=True)
    args.record.write_text(json.dumps(results,indent=2))
    args.record.chmod(0o600)


if __name__=='__main__':
    main()
