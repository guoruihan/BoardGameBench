"""Turn ownership and unchanged duo transitions through the browser adapter."""
from copy import deepcopy
import hashlib
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pytest.importorskip("torch")
from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.ui.duo import DuoPlaySession
from boardbench.ui.server import make_server


@pytest.fixture
def config(tmp_path, monkeypatch):
    class Solver:
        def __init__(self, device):
            assert device == "cpu"

        def load_weights(self, path):
            pass

        def start_episode(self, observation):
            pass

        def decide(self, observation):
            return observation["legal_actions"][0]

    monkeypatch.setattr("boardbench.v145.policy.SelfplaySolver", Solver)
    model = tmp_path / "model.pt"
    model.write_bytes(b"test model")
    return {"task": {"id": HarmoniesDuo.task_id, "version": HarmoniesDuo.rules_version},
            "model_path": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}


@pytest.mark.parametrize("seat", [0, 1])
def test_full_game_turns_and_reference_parity(config, seat):
    session = DuoPlaySession(config)
    session.new_game(seed=777, human_seat=seat)
    reference = HarmoniesDuo()
    observation, _ = reference.reset(777)
    while not session.observation["terminated"]:
        while not observation["terminated"] and observation["active_player"] != seat:
            observation = reference.step(observation["legal_actions"][0]).observation
        assert session.observation == observation
        snapshot = session.env.save_state()
        with pytest.raises(InvalidAction):
            session.step({"type": "stop_episode"})
        assert session.env.save_state() == snapshot
        action = observation["legal_actions"][0]
        observation = reference.step(action).observation
        session.step(action)
    while not observation["terminated"]:
        observation = reference.step(observation["legal_actions"][0]).observation
    assert session.observation == observation
    turns = [p["completed_turns"] for p in observation["players"]]
    assert turns[0] == turns[1]
    assert session.env.validate()
    with pytest.raises(InvalidAction, match="finished"):
        session.step({"type": "end_turn"})


def test_http_minimal_duo_routes(config):
    server = make_server(config=config, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, value):
        return json.load(urlopen(Request(base+path, json.dumps(value).encode(),
                                         {"Content-Type": "application/json"})))

    try:
        assert "双人对战" in urlopen(base).read().decode()
        state = post("/api/new", {"human_seat": 1, "seed": "9007199254740993"})
        assert state["seed"] == "9007199254740993"
        assert state["observation"]["active_player"] == state["human_seat"] == 1
        assert state["observation"]["players"][0]["completed_turns"] == 1
        assert len(state["observation"]["token_market"]) == 5
        assert not {"private", "snapshot", "frames", "bag", "deck"} & state.keys()
        assert not {"private", "bag", "deck"} & state["observation"].keys()
        before = deepcopy(state)
        with pytest.raises(HTTPError) as error:
            post("/api/new", {"human_seat": 2})
        assert error.value.code == 400
        assert json.load(urlopen(base+"/api/state")) == before
        post("/api/action", {"action": state["observation"]["legal_actions"][0]})
        with pytest.raises(HTTPError) as error:
            post("/api/save", {})
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_model_hash_is_pinned(config):
    config["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        DuoPlaySession(config)
