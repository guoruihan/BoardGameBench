"""Linux namespace isolation and explicitly *sampled* pilot resource enforcement.

Not a cgroup quota: CPU is a lower bound for adversarial auto-reaped children;
RSS sums resident pages (shared pages count repeatedly), misses between-sample
peaks and excludes file cache. Normal waited descendants are included by wait4.
The trusted ledger must live outside every writable sandbox mount.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from typing import Callable

WRAPPER = Path(os.environ.get('BOARDBENCH_CODEX_WRAPPER', shutil.which('codex') or 'codex'))
ACCOUNTING = ('approximate sampled process-tree CPU with final wait4; snapshots can race; undercounts if children '
              'are auto-reaped without wait; sampled summed RSS, shared pages '
              'double counted and transient peaks/file cache excluded; no cgroup quota')


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float
    wall_seconds: float
    rss_bytes: int
    sample_seconds: float = 0.05

    def validate(self):
        for value in (self.cpu_seconds, self.wall_seconds, self.rss_bytes, self.sample_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('All resource limits must be finite and positive')
        if self.sample_seconds > 1:
            raise ValueError('Sampling interval must be <= 1 second')


@dataclass(frozen=True)
class RuntimeMount:
    source: Path
    target: str


@dataclass
class SandboxSpec:
    argv: list[str]
    work_dir: Path
    sdk_dir: Path
    ledger_path: Path
    cpu_ids: tuple[int, ...]
    limits: Limits
    runtime_mounts: list[RuntimeMount] = field(default_factory=list)
    # socket name -> host Unix socket; HTTP TCP ports map only to these sockets.
    sockets: dict[str, Path] = field(default_factory=dict)
    tcp_bridges: dict[int, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    stdin: bytes = b''
    require_kernel_accounting: bool = False
    stop_requested: Callable[[], str | None] | None = None


@dataclass
class SandboxResult:
    exit_code: int | None
    reason: str
    cpu_seconds: float
    peak_rss_bytes: int
    active_wall_seconds: float
    cleanup_verified: bool
    accounting: str
    sample_seconds: float
    termination_latency_seconds: float
    stdout_path: str
    stderr_path: str
    max_sampling_gap_seconds: float
    cpu_overshoot_seconds: float
    wall_overshoot_seconds: float
    rss_overshoot_bytes: int


def physical_cpu_ids(count: int) -> tuple[int, ...]:
    """One available logical CPU per package/core; record these IDs in metadata."""
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((topology / 'physical_package_id').read_text().strip(),
               (topology / 'core_id').read_text().strip())
        if key not in seen:
            seen.add(key)
            selected.append(cpu)
    if count <= 0 or len(selected) < count:
        raise ValueError('Insufficient distinct physical cores in current affinity')
    return tuple(selected[:count])


def _magic(path):
    with Path(path).open('rb') as stream:
        return stream.read(4)


def discover_codex_runtime(wrapper: Path = WRAPPER) -> list[RuntimeMount]:
    """Inspect wrapper statically, never execute personal bootstrap on the host.

    Mount its real native ELF at REAL_CODEX; omit the Node dispatch layer. The
    wrapper source home is absent, so its optional personal links cannot exist.
    """
    text = wrapper.read_text()
    match = re.search(r'^REAL_CODEX="([^"$]+)"$', text, re.M)
    if not match:
        raise RuntimeError('Unsupported wrapper: literal REAL_CODEX required')
    launcher = Path(match[1])
    package = launcher.resolve().parent.parent
    binaries = list((package / 'node_modules/@openai/codex-linux-x64/vendor').glob('*/bin/codex'))
    if len(binaries) != 1 or _magic(binaries[0]) != b'\x7fELF':
        raise RuntimeError('Cannot uniquely discover native Codex ELF')
    code_mode_host = binaries[0].with_name('codex-code-mode-host')
    if not code_mode_host.is_file() or _magic(code_mode_host) != b'\x7fELF':
        raise RuntimeError('Native Codex code-mode host ELF is required')
    return [RuntimeMount(wrapper, str(wrapper)), RuntimeMount(binaries[0], str(launcher)),
            RuntimeMount(code_mode_host, str(launcher.with_name('codex-code-mode-host')))]


def minimal_runtime(include_codex: bool = False, include_compiler: bool = False) -> list[RuntimeMount]:
    """Curated system commands, system stdlib and their ELF dependencies only.

    No /usr, virtualenv, home, repository or site-packages directory wholesale.
    Additional packages require an explicitly reviewed RuntimeMount list.
    """
    mounts = {}
    binaries = []
    for name in ('python3', 'bash', 'sh', 'env', 'mkdir', 'date', 'readlink',
                 'git', 'ls', 'cat', 'cp', 'mv', 'rm', 'ln', 'basename', 'dirname', 'touch', 'head', 'tail',
                 'wc', 'sed', 'grep', 'find', 'sort', 'sleep', 'true'):
        path = Path('/usr/bin') / name
        if path.exists():
            mounts[str(path)] = RuntimeMount(path.resolve(), str(path))
            binaries.append(path)
    version = subprocess.check_output(['/usr/bin/python3', '-I', '-c',
                                      'import sys;print("%d.%d"%sys.version_info[:2])'], text=True).strip()
    stdlib = Path('/usr/lib') / f'python{version}'
    mounts[str(stdlib)] = RuntimeMount(stdlib, str(stdlib))
    binaries.extend(stdlib.glob('lib-dynload/*.so'))
    locale = Path('/usr/lib/locale/C.UTF-8')
    if locale.is_dir():
        mounts[str(locale)] = RuntimeMount(locale, str(locale))
    if include_compiler:
        for mount in compiler_runtime():
            mounts[mount.target] = mount
            if mount.source.is_file() and _magic(mount.source) == b'\x7fELF':
                binaries.append(mount.source)
    if include_codex:
        for mount in discover_codex_runtime():
            mounts[mount.target] = mount
            if _magic(mount.source) == b'\x7fELF':
                binaries.append(mount.source)
    # glibc loads this on demand for pthread cancellation/unwinding. It is
    # absent from ldd output; Python bridge-thread and Codex teardown need it.
    unwind = Path('/lib/x86_64-linux-gnu/libgcc_s.so.1')
    if not unwind.is_file():
        raise RuntimeError('Sandbox threads require the libgcc_s unwinding runtime')
    mounts[str(unwind)] = RuntimeMount(unwind.resolve(), str(unwind))
    binaries.append(unwind)
    for path in binaries:
        output = subprocess.run(['/usr/bin/ldd', str(path)], capture_output=True,
                                text=True, env={'PATH': '/usr/bin:/bin'}).stdout
        for dependency in re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)', output, re.M):
            source = Path(dependency)
            mounts[dependency] = RuntimeMount(source.resolve(), dependency)
    return list(mounts.values())


def compiler_runtime() -> list[RuntimeMount]:
    """System GCC C/C++ toolchain, with explicitly scoped headers/link inputs.

    Supports normal shared-library C/C++ builds, including pthreads and LTO;
    does not promise arbitrary external development packages or static glibc.
    """
    compiler = Path('/usr/bin/gcc')
    if not compiler.is_file():
        raise RuntimeError('System GCC is required for include_compiler')
    def query(argument):
        return subprocess.check_output([str(compiler), argument], text=True,
                                       env={'PATH': '/usr/bin:/bin'}).strip()
    machine, version = query('-dumpmachine'), query('-dumpversion')
    if machine != 'x86_64-linux-gnu':
        raise RuntimeError('Only the x86_64-linux-gnu compiler runtime is validated')
    root = Path(query('-print-file-name=crtbegin.o')).parent
    paths = [Path('/usr/bin')/name for name in ('gcc', 'g++', 'cc', 'c++', 'as', 'ld', 'ar', 'ranlib')]
    paths += [root/name for name in ('cc1', 'cc1plus', 'collect2', 'lto-wrapper', 'lto1',
              'include', 'libgcc.a', 'libgcc_eh.a', 'libgcc_s.so', 'libstdc++.so',
              'libstdc++.a', 'libsupc++.a', 'libstdc++fs.a', 'liblto_plugin.so')]
    paths += list(root.glob('crt*.o'))
    paths += list(Path('/usr/include').glob('*.h'))
    paths += [Path('/usr/include')/name for name in ('linux', 'asm-generic', 'c++/'+version)]
    headers = Path('/usr/include')/machine
    paths += list(headers.glob('*.h'))
    paths += [headers/name for name in ('asm', 'bits', 'gnu', 'sys', 'c++/'+version)]
    system_lib = Path('/usr/lib')/machine
    paths += [system_lib/name for name in ('crt1.o', 'Scrt1.o', 'crti.o', 'crtn.o',
              'libc.so', 'libc_nonshared.a', 'libm.so', 'libpthread.so', 'libdl.so',
              'librt.so', 'libstdc++.so.6', 'libgcc_s.so.1')]
    paths += [Path('/lib')/machine/name for name in ('ld-linux-x86-64.so.2', 'libmvec.so.1')]
    # GCC may dispatch using the target-prefixed assembler/linker paths.
    paths += [Path('/usr/bin')/(machine+'-'+name) for name in ('as', 'ld')]
    mounts = {}
    for path in paths:
        if not path.exists():
            raise RuntimeError('Required C/C++ runtime component missing: '+str(path))
        mounts[str(path)] = RuntimeMount(path.resolve(), str(path))
    return list(mounts.values())


def runtime_inventory(mounts: list[RuntimeMount]) -> list[dict]:
    """Hash each mounted file or directory tree for trusted provenance records."""
    result = []
    for mount in sorted(mounts, key=lambda value: value.target):
        source = Path(mount.source).resolve()
        digest = hashlib.sha256()
        files, size = 0, 0
        members = sorted(source.rglob('*')) if source.is_dir() else [source]
        for member in members:
            name = member.relative_to(source).as_posix() if source.is_dir() else '.'
            if member.is_symlink():
                digest.update(b'L'+name.encode()+b'\0'+os.readlink(member).encode()+b'\0')
            elif member.is_file():
                files += 1
                size += member.stat().st_size
                digest.update(b'F'+name.encode()+b'\0'+str(member.stat().st_size).encode()+b'\0')
                with member.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024**2), b''):
                        digest.update(chunk)
        result.append({'source': str(source), 'target': mount.target,
                       'kind': 'directory' if source.is_dir() else 'file',
                       'files': files, 'bytes': size, 'sha256': digest.hexdigest()})
    return result


# PID 1 remains alive until all adopted children have been waited. A parent that
# exits leaving background jobs does not end a run or stop resource charging.
_INIT = r'''
import json, os, select, socket, subprocess, sys, threading
config=json.load(open('/runtime/config.json'))
def bridge(listener, path):
    def connection(client):
        upstream=socket.socket(socket.AF_UNIX)
        try:
            upstream.connect(path)
            while True:
                ready,_,_=select.select([client,upstream],[],[])
                for src in ready:
                    data=src.recv(65536)
                    if not data:return
                    (upstream if src is client else client).sendall(data)
        finally:
            client.close();upstream.close()
    while True:
        client,_=listener.accept()
        threading.Thread(target=connection,args=(client,),daemon=True).start()
for port,name in config['bridges'].items():
    listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind(('127.0.0.1',int(port)));listener.listen(32)
    threading.Thread(target=bridge,args=(listener,'/run/bench/'+name),daemon=True).start()
child=subprocess.Popen(config['argv'],close_fds=True)
code=0
while True:
    try:pid,status,usage=os.wait4(-1,0)
    except ChildProcessError:break
    if pid==child.pid:code=os.waitstatus_to_exitcode(status) if hasattr(os,'waitstatus_to_exitcode') else (os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status))
sys.exit(code if code>=0 else 128-code)
'''


def _seccomp(path: Path):
    if platform.machine() != 'x86_64':
        raise RuntimeError('Only x86_64 seccomp profile has been validated')
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E),
                    (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0),
                    (0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)]
    # clone3 stores flags behind a pointer, which classic seccomp cannot inspect.
    # ENOSYS requests libc/runtime fallback to clone; ordinary thread/fork clone
    # remains usable, while all namespace-creation flags are explicitly denied.
    instructions.extend([(0x15, 0, 1, 435), (0x06, 0, 0, 0x00050026),
                         (0x15, 0, 3, 56), (0x20, 0, 0, 16),
                         (0x45, 0, 1, 0x7e020000), (0x06, 0, 0, 0x00050001),
                         (0x20, 0, 0, 0)])
    # Prevent affinity widening and nested mounts/namespaces. PID namespace
    # contains setsid/process-group escapes without forbidding ordinary shells.
    for syscall in (165, 166, 203, 272, 308):
        instructions.extend([(0x15, 0, 1, syscall), (0x06, 0, 0, 0x00050001)])
    instructions.append((0x06, 0, 0, 0x7fff0000))
    path.write_bytes(b''.join(struct.pack('HBBI', *row) for row in instructions))


def _tree_sample(root: int):
    """Snapshot approximate: /proc changes during traversal; final wait4 reconciles."""
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
            fields = raw[raw.rindex(')') + 2:].split()
            rows[int(entry.name)] = (int(fields[1]), int(fields[11]) + int(fields[12])
                                    + int(fields[13]) + int(fields[14]),
                                    int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            continue  # unrelated processes may be protected by hidepid
    included = {root}
    while True:
        new = {pid for pid, row in rows.items() if row[0] in included}
        if new <= included:
            break
        included |= new
    cpu = sum(rows[pid][1] for pid in included if pid in rows) / os.sysconf('SC_CLK_TCK')
    rss = sum(rows[pid][2] for pid in included if pid in rows)
    identities = {pid: rows[pid][3] for pid in included if pid in rows}
    return cpu, rss, identities


def _alive(pid, start):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        return fields[19] == start and fields[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class SandboxSupervisor:
    def run(self, spec: SandboxSpec) -> SandboxResult:
        spec.limits.validate()
        if spec.require_kernel_accounting:
            raise RuntimeError('Kernel-hard accounting unavailable: cgroup delegation required')
        if not spec.argv or not spec.cpu_ids or not set(spec.cpu_ids) <= os.sched_getaffinity(0):
            raise ValueError('Command and available CPU affinity are required')
        cores = []
        for cpu in spec.cpu_ids:
            base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            cores.append(((base/'physical_package_id').read_text(), (base/'core_id').read_text()))
        if len(set(cores)) != len(cores):
            raise ValueError('CPU IDs must select distinct physical cores')
        work, sdk, ledger_path = (Path(p).resolve() for p in
                                  (spec.work_dir, spec.sdk_dir, spec.ledger_path))
        if not work.is_dir() or not sdk.is_dir():
            raise ValueError('Existing work and SDK directories required')
        for exposed in (work, sdk, *(Path(m.source).resolve() for m in spec.runtime_mounts)):
            if ledger_path == exposed or exposed in ledger_path.parents:
                raise ValueError('Ledger must not be exposed to the sandbox')
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
                'cpu_seconds': 0., 'active_wall_seconds': 0., 'runs': []}
            if ledger.get('running'):
                raise RuntimeError('Interrupted ledger requires explicit trusted reconciliation; no free restart')
            return self._run(spec, work, sdk, ledger_path, ledger)

    def _run(self, spec, work, sdk, ledger_path, ledger):
        bubblewrap = shutil.which('bwrap')
        if not bubblewrap:
            raise RuntimeError('bubblewrap required; no unsandboxed fallback')
        run_number = len(ledger['runs'])
        stdout_path = ledger_path.parent / f'{ledger_path.stem}.{run_number}.stdout'
        stderr_path = ledger_path.parent / f'{ledger_path.stem}.{run_number}.stderr'
        if ledger['cpu_seconds'] >= spec.limits.cpu_seconds or ledger['active_wall_seconds'] >= spec.limits.wall_seconds:
            raise RuntimeError('Persistent cumulative resource budget exhausted')
        with tempfile.TemporaryDirectory(prefix='pilot-supervisor-') as temp:
            trusted = Path(temp)
            (trusted/'init.py').write_text(_INIT)
            (trusted/'stdin').write_bytes(spec.stdin)
            (trusted/'config.json').write_text(json.dumps({'argv': spec.argv, 'bridges': spec.tcp_bridges}))
            _seccomp(trusted/'filter')
            command = [bubblewrap, '--unshare-all', '--as-pid-1', '--die-with-parent',
                       '--cap-drop', 'ALL', '--clearenv', '--new-session',
                       '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                       '--dir', '/run/bench', '--symlink', 'usr/bin', '/bin',
                       '--bind', str(work), '/work', '--ro-bind', str(sdk), '/bench',
                       '--ro-bind', str(trusted), '/runtime', '--chdir', '/work']
            mounts = spec.runtime_mounts or minimal_runtime()
            for mount in mounts:
                target = Path(mount.target)
                if not target.is_absolute() or '..' in target.parts or str(target) in ('/', '/usr', '/home', '/dev', '/proc', '/sys'):
                    raise ValueError('Runtime mounts must be narrowly scoped absolute paths')
                if target.parts[1] in ('work', 'bench', 'runtime', 'run', 'tmp'):
                    raise ValueError('Runtime mount overlaps protected sandbox path')
                command += ['--ro-bind', str(Path(mount.source).resolve()), str(target)]
            for name, source in spec.sockets.items():
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not Path(source).is_socket():
                    raise ValueError('Only named existing Unix service sockets may be exposed')
                command += ['--bind', str(Path(source).resolve()), '/run/bench/' + name]
            if any(name not in spec.sockets or not 1024 <= int(port) <= 65535
                   for port, name in spec.tcp_bridges.items()):
                raise ValueError('Bridge ports must map to declared sockets')
            env = {'HOME': '/work', 'CODEX_HOME': '/work/.codex', 'CODEX_RUNTIME_HOME': '/work/.codex',
                   'CODEX_SQLITE_DIR': '/work/.sqlite', 'PATH': '/usr/bin:/bin',
                   'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': str(len(spec.cpu_ids)), 'OPENBLAS_NUM_THREADS': str(len(spec.cpu_ids))}
            if any(key in env and value != env[key] for key, value in spec.env.items()):
                raise ValueError('Cannot override sandbox isolation environment')
            env.update(spec.env)
            for key, value in env.items():
                command += ['--setenv', key, value]
            with (trusted/'filter').open('rb') as filt, (trusted/'stdin').open('rb') as input_file, stdout_path.open('wb') as out, stderr_path.open('wb') as err:
                command += ['--seccomp', str(filt.fileno()), '/usr/bin/python3', '-I', '/runtime/init.py']
                ledger['running'] = True
                _write_json(ledger_path, ledger)
                started = time.monotonic()
                process = None
                last_sample, max_gap = started, 0.
                reason, peak, sampled_cpu, killed_at, observed = 'exited', 0, 0., None, {}
                try:
                    # taskset executes before bwrap installs the affinity-denying filter.
                    process = subprocess.Popen(['/usr/bin/taskset', '-c', ','.join(map(str, spec.cpu_ids)), *command],
                                               stdin=input_file, stdout=out, stderr=err, pass_fds=(filt.fileno(),))
                    usage = None
                    while True:
                        now = time.monotonic()
                        max_gap = max(max_gap, now-last_sample)
                        last_sample = now
                        cpu, rss, identities = _tree_sample(process.pid)
                        observed.update(identities)
                        sampled_cpu, peak = max(sampled_cpu, cpu), max(peak, rss)
                        waited, status, final_usage = os.wait4(process.pid, os.WNOHANG)
                        if waited:
                            usage = final_usage
                            process.returncode = os.waitstatus_to_exitcode(status)
                            break
                        wall = time.monotonic() - started
                        if killed_at is None:
                            requested = spec.stop_requested() if spec.stop_requested else None
                            if requested:
                                reason = str(requested)
                            elif ledger['cpu_seconds'] + sampled_cpu >= spec.limits.cpu_seconds:
                                reason = 'cpu_limit'
                            elif rss >= spec.limits.rss_bytes:
                                reason = 'rss_limit'
                            elif ledger['active_wall_seconds'] + wall >= spec.limits.wall_seconds:
                                reason = 'wall_limit'
                            if reason != 'exited':
                                killed_at = time.monotonic()
                                self._kill(process.pid, observed)
                        if killed_at is not None and time.monotonic() - killed_at > 2:
                            process.kill()
                        time.sleep(spec.limits.sample_seconds)
                    if usage is not None:
                        sampled_cpu = max(sampled_cpu, usage.ru_utime + usage.ru_stime)
                    if reason == 'exited' and ledger['cpu_seconds'] + sampled_cpu >= spec.limits.cpu_seconds:
                        reason = 'cpu_limit'
                    if reason == 'exited' and ledger['active_wall_seconds'] + time.monotonic()-started >= spec.limits.wall_seconds:
                        reason = 'wall_limit'
                    if reason == 'exited' and process.returncode:
                        reason = 'execution_error'
                except BaseException:
                    if process is not None and process.returncode is None:
                        self._kill(process.pid, observed)
                        _, status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(status)
                        sampled_cpu = max(sampled_cpu, usage.ru_utime + usage.ru_stime)
                    # Keep running=true: interruption cannot silently reset budgets.
                    raise
                finally:
                    if process is not None:
                        self._kill(process.pid, observed)
                finished = time.monotonic()
                cleanup = not any(_alive(pid, start) for pid, start in observed.items())
                result = SandboxResult(process.returncode, reason, sampled_cpu, peak,
                                       finished-started, cleanup, ACCOUNTING, spec.limits.sample_seconds,
                                       0. if killed_at is None else finished-killed_at,
                                       str(stdout_path), str(stderr_path), max_gap,
                                       max(0., ledger['cpu_seconds'] + sampled_cpu-spec.limits.cpu_seconds),
                                       max(0., ledger['active_wall_seconds'] + finished-started-spec.limits.wall_seconds),
                                       max(0, peak-spec.limits.rss_bytes))
                ledger['cpu_seconds'] += result.cpu_seconds
                ledger['active_wall_seconds'] += result.active_wall_seconds
                ledger['running'] = False
                ledger['runs'].append(asdict(result))
                _write_json(ledger_path, ledger)
                if not cleanup:
                    raise RuntimeError('Sandbox cleanup unverified; stop queue and inspect ledger')
                return result

    @staticmethod
    def _kill(root, observed):
        # Killing namespace PID 1 destroys all members, including double-forked
        # and setsid descendants; observed host PIDs provide a checked fallback.
        if not observed:
            _, _, current = _tree_sample(root)
            observed.update(current)
        descendants = {pid: start for pid, start in observed.items() if pid != root and _alive(pid, start)}
        targets = descendants or {pid: start for pid, start in observed.items() if pid == root}
        for pid, start in targets.items():
            if _alive(pid, start):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
