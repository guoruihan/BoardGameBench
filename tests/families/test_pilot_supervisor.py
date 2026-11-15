"""Real Linux namespace gates; no model calls or opponent weights."""
import json
import os
from pathlib import Path
import socket
import threading

import pytest

from boardbench.families.pilot_supervisor import (
    Limits, SandboxSpec, SandboxSupervisor, minimal_runtime, physical_cpu_ids,
    discover_codex_runtime, WRAPPER,
)


@pytest.fixture(scope='module')
def runtime():
    return minimal_runtime()


@pytest.fixture
def spec(tmp_path, runtime):
    work, sdk = tmp_path/'work', tmp_path/'sdk'
    work.mkdir(); sdk.mkdir()
    return SandboxSpec(['/usr/bin/python3', '-c', 'print("ok")'], work, sdk,
                       tmp_path/'trusted'/'ledger.json', physical_cpu_ids(1),
                       Limits(10, 10, 256*1024**2), runtime_mounts=runtime)


def run_code(spec, code):
    spec.argv = ['/usr/bin/python3', '-c', code]
    result = SandboxSupervisor().run(spec)
    assert result.cleanup_verified
    return result


def test_files_network_gpu_and_affinity(spec):
    secret = spec.work_dir.parent/'secret'
    secret.write_text('DO_NOT_READ')
    result = run_code(spec, f'''
import os,socket,pathlib,json
assert not pathlib.Path({str(secret)!r}).exists()
assert not pathlib.Path('/datapool/data3/storage/ruihan/code/boardgame').exists()
assert not pathlib.Path('/dev/nvidia0').exists()
assert not pathlib.Path('/sys').exists()
assert not pathlib.Path('/home/ruihan').exists()
assert sorted(os.sched_getaffinity(0))=={list(spec.cpu_ids)!r}
try:os.sched_setaffinity(0,range(256))
except PermissionError:pass
else:raise AssertionError('affinity escape')
s=socket.socket();s.settimeout(.2)
assert s.connect_ex(('1.1.1.1',443)) != 0
pathlib.Path('/work/result').write_text('writable')
try:pathlib.Path('/bench/no').write_text('bad')
except OSError:pass
else:raise AssertionError('SDK writable')
print('isolated')
''')
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    assert Path(result.stdout_path).read_text().strip() == 'isolated'
    assert (spec.work_dir/'result').read_text() == 'writable'


def test_waited_children_accounting(spec):
    result = run_code(spec, '''
import subprocess,sys
code='import time; t=time.process_time(); exec("while time.process_time()-t < .15: pass")'
for i in range(6):subprocess.run([sys.executable,'-c',code],check=True)
''')
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    assert result.cpu_seconds >= .9
    assert result.cpu_seconds < 3


def test_cpu_limit_and_double_fork_cleanup(spec):
    spec.limits = Limits(.5, 10, 256*1024**2, .02)
    result = run_code(spec, '''
import os
open('/work/pid_namespace','w').write(os.readlink('/proc/self/ns/pid'))
for i in range(3):
 if os.fork()==0:
  os.setsid()
  if os.fork():os._exit(0)
  while True:pass
while True:pass
''')
    assert result.reason == 'cpu_limit'
    assert result.cpu_seconds >= .5
    assert result.termination_latency_seconds < 2
    # Independent host-side evidence: no process retains the terminated PID NS.
    namespace = (spec.work_dir/'pid_namespace').read_text()
    for process in Path('/proc').iterdir():
        if process.name.isdigit():
            try:
                assert os.readlink(process/'ns/pid') != namespace
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass


def test_memory_limit(spec):
    spec.limits = Limits(10, 10, 64*1024**2, .02)
    result = run_code(spec, 'import time; x=bytearray(100*1024**2);time.sleep(5)')
    assert result.reason == 'rss_limit'
    assert result.peak_rss_bytes >= spec.limits.rss_bytes


def test_wall_and_external_resume_ledger(spec):
    spec.limits = Limits(10, .4, 256*1024**2, .02)
    first = run_code(spec, 'import time;time.sleep(.1)')
    assert first.reason == 'exited'
    second = run_code(spec, 'import time;time.sleep(10)')
    assert second.reason == 'wall_limit'
    assert second.active_wall_seconds < .4
    with pytest.raises(RuntimeError, match='budget exhausted'):
        SandboxSupervisor().run(spec)
    data = json.loads(spec.ledger_path.read_text())
    assert len(data['runs']) == 2
    assert data['active_wall_seconds'] >= .4


def test_interrupted_ledger_blocks_free_resume(spec):
    spec.ledger_path.parent.mkdir()
    spec.ledger_path.write_text(json.dumps({'running': True}))
    with pytest.raises(RuntimeError, match='reconciliation'):
        SandboxSupervisor().run(spec)


def test_kernel_strict_refusal(spec):
    spec.require_kernel_accounting = True
    with pytest.raises(RuntimeError, match='Kernel-hard'):
        SandboxSupervisor().run(spec)


def test_unix_bridge_only(spec):
    path = spec.work_dir.parent/'gateway.sock'
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path));server.listen(1)
    def reply():
        client,_=server.accept()
        with client:
            assert client.recv(32) == b'hello'
            client.sendall(b'world')
        server.close()
    worker=threading.Thread(target=reply,daemon=True);worker.start()
    spec.sockets={'gateway.sock':path};spec.tcp_bridges={18080:'gateway.sock'}
    result = run_code(spec, '''
import socket
s=socket.create_connection(('127.0.0.1',18080));s.sendall(b'hello');assert s.recv(32)==b'world'
''')
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_clean_wrapper_version(spec):
    if not WRAPPER.exists():
        pytest.skip('Local native Codex wrapper absent')
    spec.runtime_mounts = minimal_runtime(include_codex=True)
    spec.argv = [str(WRAPPER), '--version']
    result = SandboxSupervisor().run(spec)
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    assert 'codex' in Path(result.stdout_path).read_text().lower()
    assert not (spec.work_dir/'.codex'/'auth.json').exists()
    assert not (spec.work_dir/'.codex'/'AGENTS.md').exists()


def test_external_gt_stop_and_large_unused_stdin(spec):
    spec.stdin = b'x' * (1024 * 1024)
    spec.stop_requested = lambda: 'gt_completed'
    result = run_code(spec, 'import time;time.sleep(10)')
    assert result.reason == 'gt_completed'
    assert result.active_wall_seconds < 2
    assert result.max_sampling_gap_seconds >= 0


def test_native_code_mode_spawn_and_thread_cancellation(spec):
    """Exercise worker startup and dlopen dependency omitted by a --version gate."""
    import shutil
    import subprocess
    compiler = shutil.which('cc')
    if not WRAPPER.exists() or compiler is None:
        pytest.skip('Native Codex and C compiler are required for this integration gate')
    spec.runtime_mounts = minimal_runtime(include_codex=True)
    source = spec.work_dir/'cancel.c'
    source.write_text(r'''
#include <pthread.h>
#include <unistd.h>
static void *worker(void *unused) { for (;;) { usleep(1000); } }
int main(void) {
    pthread_t thread; void *result;
    if (pthread_create(&thread, 0, worker, 0)) return 1;
    if (pthread_cancel(thread)) return 2;
    if (pthread_join(thread, &result)) return 3;
    return result == PTHREAD_CANCELED ? 0 : 4;
}
''')
    subprocess.run([compiler, '-pthread', str(source), '-o', str(spec.work_dir/'cancel')], check=True)
    host = next(m.target for m in spec.runtime_mounts if m.target.endswith('/codex-code-mode-host'))
    result = run_code(spec, f'''
import subprocess,ctypes
ctypes.CDLL('libgcc_s.so.1')
subprocess.run(['/work/cancel'],check=True)
subprocess.run([{host!r}, '--listen','stdio'],input=b'',check=True,timeout=5)
print('native worker and cancellation passed')
''')
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    assert 'native worker and cancellation passed' in Path(result.stdout_path).read_text()


def test_nested_namespace_syscalls_blocked(spec):
    result = run_code(spec, """
import ctypes,errno
libc=ctypes.CDLL(None,use_errno=True)
# clone3 cannot be safely filtered by pointed-to flags: force libc fallback.
assert libc.syscall(435,0,0)==-1 and ctypes.get_errno()==errno.ENOSYS
# clone flags CLONE_NEWUSER, CLONE_NEWNS, CLONE_NEWNET, CLONE_NEWPID.
for flag in (0x10000000,0x20000,0x40000000,0x20000000):
    assert libc.syscall(56,flag|17,0,0,0,0)==-1 and ctypes.get_errno()==errno.EPERM
print('namespace flags denied')
""")
    assert result.reason == 'exited', Path(result.stderr_path).read_text()


def test_optional_cpp_compile_and_run(spec):
    from boardbench.families.pilot_supervisor import runtime_inventory
    spec.runtime_mounts = minimal_runtime(include_compiler=True)
    spec.limits = Limits(30, 30, 512*1024**2)
    (spec.work_dir/'main.cpp').write_text(r'''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
int main() {
    std::vector<int> values{5, 1, 3};
    std::thread worker([&] { std::sort(values.begin(), values.end()); });
    worker.join();
    if (values[0] != 1 || std::accumulate(values.begin(),values.end(),0) != 9) return 1;
    std::cout << "cpp worker passed " << std::sqrt(81.0) << std::endl;
}
''')
    (spec.work_dir/'main.c').write_text('#include <stdio.h>\nint main(void){puts("c passed");return 0;}\n')
    result=run_code(spec, '''
import subprocess
subprocess.run(['g++','-std=c++17','-O2','-flto','-pthread','/work/main.cpp','-o','/work/main'],check=True)
subprocess.run(['/work/main'],check=True)
subprocess.run(['gcc','-O2','/work/main.c','-o','/work/main_c'],check=True)
subprocess.run(['/work/main_c'],check=True)
''')
    assert result.reason == 'exited', Path(result.stderr_path).read_text()
    assert 'cpp worker passed 9' in Path(result.stdout_path).read_text()
    assert 'c passed' in Path(result.stdout_path).read_text()
    inventory=runtime_inventory(spec.runtime_mounts)
    assert any(row['target']=='/usr/bin/g++' and len(row['sha256'])==64 for row in inventory)
    assert not any(row['target'] in ('/usr','/home','/usr/lib') for row in inventory)
    assert not any(m.target=='/usr/bin/g++' for m in minimal_runtime())
    (spec.ledger_path.parent/'runtime_inventory.json').write_text(json.dumps(inventory,indent=2))


def test_utf8_locale_no_shell_warning(spec):
    spec.argv=['/usr/bin/bash','-c','printf "%s" "$LANG"']
    result=SandboxSupervisor().run(spec)
    assert result.reason=='exited', Path(result.stderr_path).read_text()
    assert Path(result.stdout_path).read_text()=='C.UTF-8'
    assert 'locale' not in Path(result.stderr_path).read_text().lower()
