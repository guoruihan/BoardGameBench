from copy import deepcopy
import json
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from boardbench.environments import TASKS
from boardbench.ui.server import PlaySession, Replay, make_server


@pytest.mark.parametrize("task_id", ("micro_tiles", "take_it_easy", "harmonies"))
def test_ui_full_game_hint_restore_challenge_replay(task_id, tmp_path):
    config = {"task": {"id": task_id, "version": TASKS[task_id].spec["version"]},
              "save_directory": str(tmp_path / "saves"), "policy_root": str(tmp_path / "policies")}
    session = PlaySession(config)
    session.new_game(seed=777)
    direct = TASKS[task_id].factory()
    assert direct.reset(777)[0] == session.observation
    with pytest.raises(ValueError, match="finish"):
        session.run_challenge()
    for _ in range(5):
        original = session.env.save_state()
        a = session.hint("heuristic")["action"]
        session.hint("search")
        assert session.env.save_state() == original
        assert session.step(a) == direct.step(a)
    save_id = session.save_game()
    restored = PlaySession(config)
    restored.load_game(save_id)
    assert restored.env.save_state() == session.env.save_state()
    while not session.observation["terminated"]:
        a = session.hint("heuristic")["action"]
        assert session.step(a) == direct.step(a) == restored.step(a)
    assert session.observation["score"] == session.env.score_breakdown()["total"]
    outcome = session.run_challenge("heuristic")
    assert outcome["human_score"] == outcome["bot_score"]
    assert outcome["frames"][0]["observation"] == session.frames[0]["observation"]
    replay_path = session.export_replay()
    replay = Replay(replay_path)
    assert replay.task["id"] == task_id
    for i, frame in enumerate(session.frames):
        assert replay.state(1, i)["observation"] == frame["observation"]
    with pytest.raises(ValueError):
        restored.load_game("../../escape")


def test_v1_http_and_midturn_save(tmp_path):
    config = {"task": {"id": "harmonies", "version": "harmonies_solo_a_v1"},
              "save_directory": str(tmp_path), "policy_root": str(tmp_path / "policies")}
    server = make_server(config=config, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(path, payload):
        return json.load(urlopen(Request(base+path, json.dumps(payload).encode(), {"Content-Type": "application/json"})))
    try:
        assert '策略实验室' in urlopen(base).read().decode()
        post('/api/action', {'action': {'type': 'choose_bundle', 'bundle_id': 0}})
        current = post('/api/hint', {'policy_id': 'heuristic'})
        before = deepcopy(current['observation'])
        assert sum(before['pending_tokens'].values()) == 3
        saved = post('/api/save', {})
        post('/api/bot-step', {'policy_id': 'heuristic'})
        restored = post('/api/load', {'save_id': saved['save_id']})
        assert restored['observation'] == before
        assert 'private' not in restored and 'snapshot' not in restored
        new = post('/api/new', {'task_id': 'micro_tiles', 'seed': '9007199254740993'})
        assert new['seed'] == '9007199254740993'  # Never round seed via JavaScript float.
        assert new['task']['id'] == 'micro_tiles'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
