"""Bounded, descriptor-relative copies from a concurrently untrusted directory.

The caller owns the destination parent exclusively. Source directories/files are
opened with O_NOFOLLOW at *every* path component; copied contents are validated
only after they reside in the trusted destination. No submitted code is executed.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import sys


class SnapshotError(ValueError):
    """Unsafe, unstable or oversized submitted filesystem tree."""


def _open_directory(path: Path) -> int:
    path = Path(os.path.abspath(path))
    # abspath normalizes '..', which is safe here: each resulting component is
    # still opened independently and symbolic-link ancestors remain forbidden.
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def safe_snapshot(source: Path, dest: Path, max_bytes: int = 64 * 1024**2,
                  max_files: int = 10000) -> dict[str, int]:
    """Copy regular files/directories to a new trusted directory; reject links.

    Limits include all directory entries (directories count toward max_files).
    Failure removes the incomplete destination; an existing destination is never
    overwritten or removed. A concurrent rename cannot redirect an open dirfd.
    Concurrent regular-file edits are rejected when metadata changes during copy;
    semantic consistency across files is the subsequent manifest validator's job.
    """
    if type(max_bytes) is not int or type(max_files) is not int or min(max_bytes, max_files) <= 0:
        raise ValueError('Snapshot bounds must be positive integers')
    source, dest = Path(source), Path(dest)
    source_abs, dest_abs = Path(os.path.abspath(source)), Path(os.path.abspath(dest))
    if source_abs.parts[1:2] in (('proc',), ('sys',), ('dev',)):
        raise SnapshotError('Kernel virtual filesystems cannot be submitted')
    if source_abs == dest_abs or source_abs in dest_abs.parents:
        raise SnapshotError('Destination cannot be inside source')
    src = parent = target = None
    created = False
    counters = {'bytes': 0, 'files': 0}
    source_device = None

    def copy_directory(source_fd: int, destination_fd: int, depth: int):
        if depth > 64:
            raise SnapshotError('Directory depth exceeds 64')
        for name in os.listdir(source_fd):
            counters['files'] += 1
            if counters['files'] > max_files:
                raise SnapshotError('Snapshot entry limit exceeded')
            # Metadata inspection never follows links. A substitution after this
            # inspection still has to pass the O_NOFOLLOW open and fstat below.
            before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            if before.st_dev != source_device:
                raise SnapshotError('Filesystem mount crossings are forbidden')
            if stat.S_ISDIR(before.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=source_fd)
                try:
                    if os.fstat(child).st_dev != source_device:
                        raise SnapshotError('Filesystem mount crossings are forbidden')
                    os.mkdir(name, 0o755, dir_fd=destination_fd)
                    output = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=destination_fd)
                    try:
                        copy_directory(child, output, depth + 1)
                    finally:
                        os.close(output)
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                     dir_fd=source_fd)
                try:
                    opened = os.fstat(descriptor)
                    if opened.st_dev != source_device or not stat.S_ISREG(opened.st_mode):
                        raise SnapshotError('Only regular files and directories are allowed')
                    if opened.st_size > max_bytes - counters['bytes']:
                        raise SnapshotError('Snapshot byte limit exceeded')
                    mode = 0o755 if opened.st_mode & 0o111 else 0o644
                    output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     mode, dir_fd=destination_fd)
                    try:
                        while True:
                            chunk = os.read(descriptor, min(1024**2, max_bytes - counters['bytes'] + 1))
                            if not chunk:
                                break
                            counters['bytes'] += len(chunk)
                            if counters['bytes'] > max_bytes:
                                raise SnapshotError('Snapshot byte limit exceeded')
                            view = memoryview(chunk)
                            while view:
                                written = os.write(output, view)
                                if not written:
                                    raise SnapshotError('Snapshot write made no progress')
                                view = view[written:]
                        os.fsync(output)
                    finally:
                        os.close(output)
                    after = os.fstat(descriptor)
                    if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        raise SnapshotError('Source file changed during snapshot')
                finally:
                    os.close(descriptor)
            else:
                raise SnapshotError('Links and special files are forbidden')

    try:
        src = _open_directory(source_abs)
        source_device = os.fstat(src).st_dev
        parent = _open_directory(dest_abs.parent)
        os.mkdir(dest_abs.name, 0o755, dir_fd=parent)
        created = True
        target = os.open(dest_abs.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
        copy_directory(src, target, 0)
        os.fsync(target)
        return counters
    except OSError as error:
        raise SnapshotError(f'Unsafe or unstable source filesystem: {error.strerror}') from error
    finally:
        for descriptor in (target, parent, src):
            if descriptor is not None:
                os.close(descriptor)
        # The destination parent is exclusively trusted; source cannot rename it.
        if created and sys.exc_info()[0] is not None:
            shutil.rmtree(dest_abs)
