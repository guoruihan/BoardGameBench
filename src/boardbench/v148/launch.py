"""Frozen explicitly renewed windows; inherited guarded lifecycle and node locks."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v147.launch import guard


def validate(c,role,physical,now):
    if (c['role']!=role or c['host']!=os.uname().nodename
            or not isinstance(c.get('started_unix'),(int,float))
            or not isinstance(c.get('deadline_unix'),(int,float))
            or not 0<c['deadline_unix']-c['started_unix']<=86400
            or not c['started_unix']<=now<c['deadline_unix']-1800
            or c['authorized_wall_seconds']!=c['deadline_unix']-c['started_unix']
            or not 2<=c['cpu_cores']<=len(physical)//2 or c['seconds']!=900
            or not 1<=c['cycles']<=3):
        raise ValueError('explicit renewed window,900s games and half physical CPUs required')
    if role=='learning' and (c['host']!='a226.cluster.helixon.com' or c['cpu_cores']!=57 or c['workers']!=13 or len(c['gpu_uuids'])!=4
            or len(set(c['gpu_uuids']))!=4 or set(c['gpu_uuids'])!=set(c['authorized_gpu_uuids'])):
        raise ValueError('four independent authorized GPUs /57 CPUs required')


def launch(repo,out,config,role):
    repo=Path(repo).resolve();out=Path(out).resolve();c=read_json(config);cpus=physical_cpus()
    validate(c,role,cpus,time.time())
    if os.getloadavg()[0]>len(cpus)*.5:raise RuntimeError('busy shared host; no preemption')
    for name in ('student','opponent'):
        if digest(c[name])!=c[name+'_sha256']:raise ValueError('input model hash')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','rounds','decisions','jobs','episodes','selection','confirmation'):(out/name).mkdir()
    for name in source_files(repo):
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    c['repo']=str(repo);write_json(out/'config.json',c);write_json(out/'resolved_config.json',c)
    write_json(out/'launch.json',dict(source_id=source_id(repo),source_commit=subprocess.check_output(
        ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),cpus=cpus[:c['cpu_cores']],
        config_sha256=digest(out/'config.json'),resolved_config_sha256=digest(out/'resolved_config.json'),created_unix=time.time()))
    guard(out,role,module='boardbench.v148.launch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--config');p.add_argument('--out',required=True)
    p.add_argument('--role',choices=['search','learning'],required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.execute:
        if a.role=='search':
            from .search_campaign import execute
        else:
            from .campaign import execute
        execute(a.out)
    elif a.resume:guard(Path(a.out).resolve(),a.role,module='boardbench.v148.launch')
    else:launch(a.repo,a.out,a.config,a.role)
