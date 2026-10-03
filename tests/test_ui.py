from dataclasses import asdict
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from boardbench.environments.risk import RiskCollect
from boardbench.runner.engine import run
from boardbench.ui.server import PlaySession, Replay, make_server


def test_interactive_backend_matches_real_engine():
    session = PlaySession({"seed": 123})
    direct = RiskCollect()
    assert session.observation == direct.reset(session.seed)[0]
    for action in ("DRAW", "DRAW", "BANK"):
        expected = direct.step(action)
        assert asdict(session.step(action)) == asdict(expected)
        if expected.terminated:
            break


def test_replay_reads_recorded_observations_only(tmp_path, monkeypatch):
    out = tmp_path / "run"
    run({"solver": {"id": "random"}, "checkpoint": {"every_episodes": 0}}, out)
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not instantiate engine or draw cards")
    monkeypatch.setattr(RiskCollect, "__init__", forbidden)
    replay = Replay(out / "events.jsonl")
    for line in (out / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "transition":
            assert replay.state(event["episode"], event["step"])["observation"] == event["result"]["observation"]
    with pytest.raises(ValueError):
        replay.state(999)


def test_http_play_and_replay_api(tmp_path):
    server = make_server(config={"seed": 3}, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert "风险采集" in urlopen(base).read().decode()
        initial = json.load(urlopen(base + "/api/state"))
        assert initial["observation"]["steps"] == 0
        request = Request(base + "/api/action", json.dumps({"action": "BANK"}).encode(),
                          {"Content-Type": "application/json"})
        finished = json.load(urlopen(request))
        assert finished["observation"]["terminated"]
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 400
        new = json.load(urlopen(Request(base + "/api/new", b"{}")))
        assert not new["observation"]["terminated"]
        assert new["episode"] == initial["episode"] + 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
