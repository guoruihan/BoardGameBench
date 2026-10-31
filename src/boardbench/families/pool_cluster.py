"""Launch bounded common-policy pilots on explicitly supplied private host plans.

Host/IP inventories and launch plans are local ignored artifacts. One coordinator
plus six actors use seven distinct physical cores per run; controller core is
also reserved. Historical and third-party tasks are never interrupted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import fcntl, json, os, shlex, subprocess, time
from pathlib import Path
request=json.loads(REQUEST)
root=Path(request['root']);run=Path(request['run']); release=Path(request['release'])
locks=root/'outputs/v170/locks';locks.mkdir(parents=True,exist_ok=True)
lock=(locks/(request['host']+'.lock')).open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def cpu():
    v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    return sum(v[:8]),v[3]+v[4]
a=cpu();time.sleep(1);b=cpu();busy=100*(1-(b[1]-a[1])/max(1,b[0]-a[0]))
allowed=set(os.sched_getaffinity(0))
topology=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,ONLINE'],text=True)
cores={}
for line in topology.splitlines():
    if not line or line.startswith('#'):continue
    cpu_id,core,socket,online=line.split(',')
    if online=='Y' and int(cpu_id) in allowed:cores.setdefault((socket,core),int(cpu_id))
representatives=sorted(cores.values())
limit=min(len(representatives)//2,128 if request['host'].startswith('a') else len(representatives))
needed=len(request['jobs'])*7+1
if needed>limit:raise RuntimeError('Physical half-host CPU quota exceeded')
if busy+100*needed/len(allowed)>50:raise RuntimeError('Projected whole-host CPU busy fraction too high')
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
manifests=list(root.glob('outputs/v160/bots/*/*/launch.json'))
manifests+=list(root.glob('outputs/v170/*/runs/*/*/launch.json'))
for manifest in manifests:
    try:old=json.loads(manifest.read_text())
    except Exception:continue
    if old.get('session') in sessions and set(old.get('cpu_ids',[])) & set(representatives[:needed]):
        raise RuntimeError('Own active allocation overlaps planned physical cores')
gpu_ids=sorted({j['gpu'] for j in request['jobs']})
for gpu in gpu_ids:
    out=subprocess.check_output(['nvidia-smi','-i',str(gpu),
       '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    used,total,util=map(int,out.strip().split(','))
    procs=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    if used>100 or util>5 or procs:raise RuntimeError('GPU no longer idle; defer whole node: '+str(gpu))
launched=[]
for i,job in enumerate(request['jobs']):
    directory=run/'runs'/job['game']/job['arm']
    if (directory/'launch.json').exists():raise RuntimeError('Existing launch; inspect before retry')
    config=json.loads((directory/'config.json').read_text())
    if config['source_commit']!=request['commit']:raise RuntimeError('Source identity mismatch')
    cpus=representatives[i*7:(i+1)*7]
    config['deadline_unix']=time.time()+3600
    temporary=directory/'config.json.part';temporary.write_text(json.dumps(config,indent=2));temporary.replace(directory/'config.json')
    cmd=[str(root/'.venv-train/bin/python'),'-m','boardbench.families.pool_training','--config',str(directory/'config.json')]
    script='set -uo pipefail\ncd '+shlex.quote(str(root))+'\n'
    script+='export PYTHONPATH='+shlex.quote(str(release/'src'))+'\n'
    script+='export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
    script+='export CUDA_VISIBLE_DEVICES='+str(job['gpu'])+'\n'
    script+='taskset -c '+','.join(map(str,cpus))+' timeout --signal=TERM --kill-after=10s 3590s '+shlex.join(cmd)
    script+=' 2>&1 | tee '+shlex.quote(str(directory/'train.log'))+'\n'
    script+='code=${PIPESTATUS[0]}\nprintf "%s\\n" "$code" > '+shlex.quote(str(directory/'exit_code'))+'\nexit "$code"\n'
    (directory/'run.sh').write_text(script)
    session='v170_'+run.name+'_'+job['game']+'_'+job['arm']
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(directory/'run.sh')],check=True)
    row=dict(host=request['host'],game=job['game'],arm=job['arm'],session=session,cpu_ids=cpus,
             gpu=job['gpu'],cpu_physical_quota=limit,cpu_busy_before=busy,launched_unix=time.time(),
             deadline_unix=config['deadline_unix'],source_commit=request['commit'])
    (directory/'launch.json').write_text(json.dumps(row,indent=2));launched.append(row)
print(json.dumps(launched))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--commit',required=True)
    a=p.parse_args()
    hosts={x['host'].split('.')[0]:x for x in json.loads(a.inventory.read_text())['hosts'] if x['status']=='ok'}
    plan=json.loads(a.plan.read_text())
    def launch(group):
        host=group['host']
        request=dict(group,root=str(a.root.resolve()),run=str(a.run.resolve()),
                     release=str(a.release.resolve()),commit=a.commit)
        code=REMOTE.replace('REQUEST',repr(json.dumps(request)))
        cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
             '-o','GlobalKnownHostsFile=/dev/null','-o','UserKnownHostsFile=/tmp/codex_known_hosts_cluster',
             '-o','StrictHostKeyChecking=accept-new','-o','CheckHostIP=no',
             'ruihan@'+hosts[host]['ip'],'python3 -c '+shlex.quote(code)]
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
            return dict(host=host,returncode=r.returncode,output=r.stdout,error=r.stderr)
        except subprocess.TimeoutExpired:
            return dict(host=host,returncode=None,error='Probe timeout; inspect manifests before any retry')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(launch,plan))
    (a.run/'launch_results.json').write_text(json.dumps(rows,indent=2))
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
