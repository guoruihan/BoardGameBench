import os
from pathlib import Path
import socket

import pytest

from boardbench.families.pilot_snapshot import safe_snapshot, SnapshotError


def test_regular_tree_and_executable_permissions(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'nested').mkdir()
    (source/'manifest.json').write_text('{}')
    program=source/'nested'/'run';program.write_text('hello');program.chmod(0o4755)
    dest=tmp_path/'dest'
    assert safe_snapshot(source,dest)=={'files':3,'bytes':7}
    assert (dest/'nested'/'run').read_text()=='hello'
    assert (dest/'nested'/'run').stat().st_mode & 0o7777 == 0o755


@pytest.mark.parametrize('kind',['symlink','dir_symlink','fifo','socket'])
def test_reject_links_and_special_files(tmp_path,kind):
    source=tmp_path/'source';source.mkdir();bad=source/'bad';server=None
    if kind=='symlink':bad.symlink_to('/etc/passwd')
    elif kind=='dir_symlink':bad.symlink_to('/etc',target_is_directory=True)
    elif kind=='fifo':os.mkfifo(bad)
    else:
        server=socket.socket(socket.AF_UNIX);server.bind(str(bad))
    try:
        with pytest.raises(SnapshotError):safe_snapshot(source,tmp_path/'dest')
        assert not (tmp_path/'dest').exists()
    finally:
        if server:server.close()


def test_reject_symlink_ancestor_and_existing_destination(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'child').mkdir()
    alias=tmp_path/'alias';alias.symlink_to(source,target_is_directory=True)
    with pytest.raises(SnapshotError):safe_snapshot(alias/'child',tmp_path/'dest')
    dest=tmp_path/'dest';dest.mkdir();(dest/'keep').write_text('untouched')
    with pytest.raises(SnapshotError):safe_snapshot(source,dest)
    assert (dest/'keep').read_text()=='untouched'


@pytest.mark.parametrize('limit', ['bytes','entries'])
def test_copy_limits_cleanup(tmp_path,limit):
    source=tmp_path/'source';source.mkdir();(source/'one').write_bytes(b'abcd');(source/'two').mkdir()
    kwargs={'max_bytes':3} if limit=='bytes' else {'max_files':1}
    with pytest.raises(SnapshotError):safe_snapshot(source,tmp_path/'dest',**kwargs)
    assert not (tmp_path/'dest').exists()


def test_racing_file_substitution_never_reads_symlink_target(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();victim=source/'victim';victim.write_text('safe')
    secret=tmp_path/'secret';secret.write_text('host secret')
    real_open=os.open;swapped=False
    def racing_open(path,flags,*args,**kwargs):
        nonlocal swapped
        if path=='victim' and not swapped:
            swapped=True;victim.unlink();victim.symlink_to(secret)
        return real_open(path,flags,*args,**kwargs)
    monkeypatch.setattr(os,'open',racing_open)
    with pytest.raises(SnapshotError):safe_snapshot(source,tmp_path/'dest')
    assert swapped
    assert not (tmp_path/'dest').exists()


def test_racing_directory_substitution_never_traverses_host(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();victim=source/'victim';victim.mkdir()
    secret=tmp_path/'host';secret.mkdir();(secret/'secret').write_text('host secret')
    real_open=os.open;swapped=False
    def racing_open(path,flags,*args,**kwargs):
        nonlocal swapped
        if path=='victim' and not swapped:
            swapped=True;victim.rmdir();victim.symlink_to(secret,target_is_directory=True)
        return real_open(path,flags,*args,**kwargs)
    monkeypatch.setattr(os,'open',racing_open)
    with pytest.raises(SnapshotError):safe_snapshot(source,tmp_path/'dest')
    assert swapped
    assert not (tmp_path/'dest').exists()


def test_destination_inside_source_rejected(tmp_path):
    with pytest.raises(SnapshotError):safe_snapshot(tmp_path,tmp_path/'dest')
