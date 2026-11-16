"""Managed-task lifecycle and publication gates, including real bwrap execution."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import pytest

from boardbench.families.pilot_jobs import ManagedJobs
from boardbench.families.pilot_run import (Checkpoints, NativePilotAdapter, PhaseContext, Budget,
    blank_submission, prepare_public_material, tree_hash)
from boardbench.families.pilot_supervisor import SandboxSpec, SandboxSupervisor, Limits, minimal_runtime, physical_cpu_ids


def test_publish_atomic_failure_and_immutable_entry(tmp_path, monkeypatch):
    candidate=tmp_path/'candidate';candidate.mkdir();(candidate/'play.py').write_text('print(1)')
    store=Checkpoints(tmp_path/'trusted')
    first=store.publish(candidate, 'play.py')
    assert json.loads((store.root/first/'manifest.json').read_text())['program_entry']=='play.py'
    (candidate/'play.py').write_text('print(2)')
    with pytest.raises(RuntimeError, match='budget'):
        store.publish(candidate,'play.py',admissible=lambda:False)
    assert json.loads((store.root/'latest.json').read_text())['sha256']==first
    original=Path.replace
    def fail_pointer(self, target):
        if Path(target).name=='latest.json':raise OSError('injected pointer failure')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',fail_pointer)
    with pytest.raises(OSError,match='injected'):
        store.publish(candidate,'play.py')
    assert store.freeze().name==first
    assert (store.root/first/'play.py').read_text()=='print(1)'
    assert tree_hash(store.root/first)==first
    with pytest.raises(RuntimeError,match='frozen'):store.publish(candidate,'play.py')


def test_queue_admission_and_interrupted_no_retry(tmp_path):
    manager=ManagedJobs(tmp_path)
    job=manager.start(['python3','x.py']);assert manager.active()
    manager.poll();manager.event('job_started',job['job_id'],pid=2)
    manager.event('controller_exit',exit_code=1)
    with pytest.raises(RuntimeError,match='closed'):manager.start(['python3'])
    manager.event('job_exit',job['job_id'],exit_code=1,signal=None)
    assert not manager.active()
    with pytest.raises(RuntimeError,match='restart'):ManagedJobs(tmp_path)


@pytest.fixture
def harness(tmp_path):
    work=tmp_path/'work';work.mkdir();sdk=tmp_path/'sdk'
    from boardbench.families import pilot_client
    prepare_public_material(sdk,Path(pilot_client.__file__),[])
    store=Checkpoints(tmp_path/'checkpoints');blank_submission(work/'submission');store.checkpoint(work/'submission')
    # Runtime discovery invokes host tools and can exceed this short test's
    # execution window on a busy filesystem; finish setup before starting it.
    runtime=minimal_runtime()
    budget=Budget(1,1,1000,1000,1000,10,2,10,1)
    ctx=PhaseContext('dev',work,tmp_path/'home',tmp_path/'ledger.json',budget,time.time()+10,'',store)
    manager=ManagedJobs(tmp_path/'jobs')
    adapter=object.__new__(NativePilotAdapter);adapter.installed_dependencies=set();adapter.gateway_close=None
    record={'jobs':manager,'finished':False,'publish_admissible':lambda *_:time.time()<ctx.deadline_unix}
    server,socket,token=adapter._control_server(ctx,record)
    spec=SandboxSpec(['/usr/bin/python3','/work/controller.py'],work,sdk,ctx.ledger_path,physical_cpu_ids(2),
        Limits(10,10,256*1024**2,.02),runtime_mounts=runtime,sockets={'control.sock':socket},
        env={'PYTHONPATH':'/bench','PILOT_CONTROL_SOCKET':'/run/bench/control.sock','PILOT_CONTROL_TOKEN':token},
        managed_jobs=True,on_started=lambda pid:setattr(manager,'sandbox_pid',pid))
    yield ctx,manager,spec
    server.shutdown();server.server_close();shutil.rmtree(socket.parent)


def test_job_publishes_after_controller_exit_without_registered_program(harness):
    ctx,manager,spec=harness
    (ctx.workspace/'finalize.py').write_text('''import time,pathlib,submission
 time.sleep(.3)
'''.replace('\n time','\ntime')+'''p=pathlib.Path('candidate');p.mkdir();(p/'play.py').write_text('print(42)')
print(submission.publish(directory='candidate',entrypoint='play.py'),flush=True)
''')
    (ctx.workspace/'controller.py').write_text("import jobs,sys\nprint(jobs.start(['python3','finalize.py']),flush=True)\nsys.exit(1)\n")
    result=SandboxSupervisor().run(spec)
    assert result.reason=='execution_error',Path(result.stderr_path).read_text()
    assert result.cleanup_verified
    assert manager.status('job-0001')['exit_code']==0
    events=[json.loads(x) for x in (manager.root/'events.jsonl').read_text().splitlines()]
    assert next(x['time'] for x in events if x['event']=='controller_exit') < next(x['time'] for x in events if x['event']=='job_exit')
    frozen=ctx.checkpoints.freeze()
    assert json.loads((frozen/'manifest.json').read_text())['program_entry']=='play.py'
    assert 'sha256' in manager.output('job-0001')['text']


def test_job_failure_keeps_previous_submission_and_broker_is_not_spoofable(harness):
    ctx,manager,spec=harness
    old=json.loads((ctx.checkpoints.root/'latest.json').read_text())['sha256']
    (ctx.workspace/'controller.py').write_text('''import jobs,pilot_control
try:pilot_control.request('broker_event',kind='controller_exit',exit_code=0)
except RuntimeError as e:assert 'broker_identity_denied' in str(e)
else:raise AssertionError('forged broker accepted')
jobs.start(['python3','-c','raise SyntaxError("compile failed")'])
''')
    result=SandboxSupervisor().run(spec)
    assert result.reason=='exited',Path(result.stderr_path).read_text()
    assert manager.status('job-0001')['exit_code']==1
    assert 'compile failed' in manager.output('job-0001',stream='stderr')['text']
    assert ctx.checkpoints.freeze().name==old


@pytest.mark.parametrize('limit',['cpu','wall','rss'])
def test_parallel_jobs_share_budget_and_all_die(harness,limit):
    ctx,manager,spec=harness
    if limit=='cpu':
        spec.limits=Limits(.7,10,256*1024**2,.02);code='while True:pass'
    elif limit=='wall':
        spec.limits=Limits(10,.7,256*1024**2,.02);code='import time;time.sleep(10)'
    else:
        spec.limits=Limits(10,10,150*1024**2,.02);code='import time;x=bytearray(80*1024**2);time.sleep(10)'
    (ctx.workspace/'controller.py').write_text('import jobs\n'+f"jobs.start(['python3','-c',{code!r}])\n"*2)
    result=SandboxSupervisor().run(spec)
    manager.close(result.reason)
    assert result.reason==limit+'_limit',Path(result.stderr_path).read_text()
    assert result.cleanup_verified
    assert len(manager.jobs)==2
    assert not manager.active()
    assert result.active_wall_seconds<3


def test_publish_and_freeze_serialize_on_complete_snapshot(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from boardbench.families import pilot_snapshot
    candidate=tmp_path/'candidate';candidate.mkdir();(candidate/'play.py').write_text('print(1)')
    store=Checkpoints(tmp_path/'trusted');first=store.publish(candidate,'play.py')
    (candidate/'play.py').write_text('print(2)')
    copied=threading.Event();proceed=threading.Event();original=pilot_snapshot.safe_snapshot
    def paused(*args):
        result=original(*args);copied.set();assert proceed.wait(5);return result
    monkeypatch.setattr(pilot_snapshot,'safe_snapshot',paused)
    with ThreadPoolExecutor(2) as pool:
        publishing=pool.submit(store.publish,candidate,'play.py')
        assert copied.wait(5)
        freezing=pool.submit(store.freeze)
        assert json.loads((store.root/'latest.json').read_text())['sha256']==first
        assert not freezing.done()
        proceed.set();new=publishing.result()
        assert freezing.result().name==new
        assert tree_hash(store.root/new)==new
