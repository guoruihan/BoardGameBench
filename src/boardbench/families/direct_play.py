"""Direct text decisions against frozen opponents; no model-side simulator tools.

Private API/CLI wiring is an external JSON-in/JSON-out command. This module only
hands that command a public player view, printed components, rules and the full
legal action table. Interrupted games are unknown, never automatic losses.
"""
import argparse
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import time

from .joint_pool import load_opponent, verify_pool
from .registry import game_class
from .state_io import copy_data

PROTOCOL = 'direct_text_play_v1'
LEARNED = {'neural', 'guided_rich', 'harmonies_rl'}


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def select_opponents(manifest):
    """Do not fabricate a second learned policy when the frozen pool has one."""
    chosen = {}
    for game, entry in manifest['games'].items():
        learned = [n for n in ('legacy', 'candidate') if n in entry['members']
                   and entry['members'][n]['spec']['kind'] in LEARNED]
        search = entry['members'].get('search_900')
        if not learned or not search or search['spec']['kind'] != 'terminal_search' \
                or search['spec']['seconds'] != 900:
            raise ValueError(f'Missing frozen learned/900s opponents for {game}')
        chosen[game] = learned + ['search_900']
    return chosen


def make_jobs(manifest, labels, seed_start=175110000):
    jobs = []
    for game, opponents in select_opponents(manifest).items():
        for opponent in opponents:
            for repeat in range(2):
                for seat in (0, 1):
                    for label in labels:
                        jobs.append(dict(id=f'{game}__{opponent}__{repeat}__{seat}__{label}',
                            game=game, opponent=opponent, seed=seed_start+repeat,
                            seat=seat, label=label))
    return jobs


def public_catalog(game):
    """Only fixed printed component dictionaries, never an environment instance."""
    module = importlib.import_module('boardbench.families.games.' + game)
    if hasattr(module, 'DATA'):
        data = copy_data(module.DATA)
        if game == 'patchwork':data['orientation_cells'] = copy_data(module.ORIENTATIONS)
        if game == 'calico':data['tile_encoding'] = 'tile = 6 * color + pattern; -1 empty, -2 goal slot'
        return data
    if game == 'targi':
        return {name: copy_data(getattr(module, name)) for name in
                ('TRIBES', 'GOODS', 'BORDER', 'COORDS', 'RESOURCE_NAMES')}
    if game == 'el_dorado':
        return {'cards': copy_data(module.CARDS), 'starter': copy_data(module.STARTER),
                'map_coordinates': 'Axial hex coordinates; adjacency directions: ' +
                '(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1).'}
    if game == 'harmonies':
        from boardbench.environments import harmonies as printed
        return {name: copy_data(getattr(printed, name)) for name in
                ('CARDS', 'CELLS', 'NEIGHBORS', 'STACKS', 'COLORS', 'INVENTORY')}
    raise ValueError(f'No public component catalogue for {game}')


def action_table(actions):
    """Lossless columnar encoding; original list order and every candidate remain."""
    columns = sorted({key for action in actions for key in action})
    rows = [[action.get(key) for key in columns] for action in actions]
    # A separate presence mask distinguishes absent fields from explicit nulls.
    presence = [[i for i, key in enumerate(columns) if key in action] for action in actions]
    return dict(columns=columns, rows=rows, present=presence,
                indexing='zero-based row number; absent columns are not action fields')


def request_for(observation, legal, rules, catalog, memory=''):
    obs = copy_data(observation)
    obs.pop('legal_actions', None)  # Same full list is encoded once below.
    return dict(protocol=PROTOCOL,
        task=('Play this actual game to win against the opponent. Choose one legal row directly. '
              'Do not write code, invoke tools, inspect files, simulate the engine, or seek advice. '
              'The rules describe the current fixed original-game configuration. Ignore any '
              'implementation assignments in the source rule notes. Only the current player view '
              'is available; hidden orders/hands are unknown. Return JSON only: '
              '{"action_index": integer, "memory": "brief private plan for your next decision"}. '
              'The memory limit is 2000 characters. Full current public history, when supplied by '
              'the game, is retained. Each call starts fresh with your previous private note.'),
        rules=rules, printed_components=catalog, observation=obs,
        legal_action_table=action_table(legal), memory=memory)


def parse_choice(text, size):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict) or type(data.get('action_index')) is not int \
            or not 0 <= data['action_index'] < size:
        raise ValueError('action_index must be a valid zero-based integer')
    memory = data.get('memory', '')
    if not isinstance(memory, str) or len(memory) > 2000:
        raise ValueError('memory must be text of at most 2000 characters')
    return data['action_index'], memory


def invoke(command, request, timeout, cwd):
    """No shell interpolation; terminate the whole adapter group on timeout."""
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=cwd, start_new_session=True)
    began = time.monotonic()
    try:
        out, err = p.communicate(json.dumps(request, ensure_ascii=False), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return dict(ok=False, error='adapter_timeout', seconds=time.monotonic()-began,
                    stdout=out, stderr=err)
    finally:
        try:os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:pass
    try:
        result = json.loads(out)
    except ValueError:
        result = dict(ok=False, error='invalid_adapter_json', stdout=out, stderr=err)
    if p.returncode:
        result.update(ok=False, error='adapter_exit', exit_code=p.returncode)
    result['wall_seconds'] = time.monotonic()-began
    return result


def play_job(pool, manifest, job, adapter, rules, directory, *, max_calls=600,
             max_decisions=3000, max_seconds=10800, call_timeout=165, max_prompt_chars=350000):
    directory = Path(directory)
    resume_path = directory/'resume.pkl'
    resuming = resume_path.exists()
    directory.mkdir(parents=True, exist_ok=resuming)
    cls = game_class(job['game'])
    env = cls(); env.reset(job['seed'], first_player=0)
    seat = job['seat']; opponent_seat = 1-seat
    entry = manifest['games'][job['game']]['members'][job['opponent']]
    search = entry['spec']['kind'] == 'terminal_search'
    opponent = load_opponent(pool, job['game'], job['opponent'], opponent_seat,
                             job['seed'], manifest)
    catalog = public_catalog(job['game'])
    began = time.monotonic(); cpu = time.process_time()
    calls = invalid = forced = 0; decisions = [0, 0]; memory = ''
    policy_seconds = [0., 0.]; status = 'running'
    metadata = dict(protocol=PROTOCOL, job=job, pool_sha256=manifest['manifest_sha256'],
        opponent_policy_id=entry['policy_id'], rules_sha256=digest(rules),
        catalog_sha256=digest(catalog), rules_version=str(cls.rules_version),
        adapter_sha256=digest(adapter),
        config=cls.config_id, first_player=0, started_unix=time.time(),
        limits=dict(max_calls=max_calls, max_decisions=max_decisions, max_seconds=max_seconds,
                    call_timeout=call_timeout, max_prompt_chars=max_prompt_chars))
    if resuming:
        previous = json.loads((directory/'metadata.json').read_text())
        for field in ('protocol','job','pool_sha256','opponent_policy_id','rules_sha256',
                      'catalog_sha256','rules_version','config','adapter_sha256','limits'):
            if previous[field] != metadata[field]:raise ValueError('Resume identity changed: '+field)
        receipt = json.loads((directory/'resume.sha256.json').read_text())
        blob = resume_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != receipt['sha256']:raise ValueError('Resume checksum mismatch')
        # Only load trusted coordinator-owned checkpoints, never submitted files.
        saved = pickle.loads(blob)
        env.restore(saved['snapshot']); opponent = saved['opponent']
        calls,invalid,forced,decisions,memory,policy_seconds = saved['counters']
        began -= saved['wall_seconds']; cpu -= saved['cpu_seconds']
        metadata = previous
        resume_path.unlink()
    dump(directory/'metadata.json', metadata)
    with (directory/'trajectory.jsonl').open('a', buffering=1) as trajectory:
        try:
            while not env.outcome()['terminated']:
                outcome = env.outcome()
                if outcome.get('unresolved') or outcome.get('unresolved_rule'):
                    status = 'unresolved_rule'; break
                if calls >= max_calls or sum(decisions) >= max_decisions \
                        or time.monotonic()-began >= max_seconds:
                    status = 'truncated'; break
                actor = env.current_player
                observation = env.observe(actor); legal = env.legal_actions(actor)
                if not legal:
                    status = 'no_legal_action'; break
                before = time.monotonic()
                if actor != seat:
                    action = opponent.decide(env) if search else opponent.decide(observation, copy_data(legal))
                elif len(legal) == 1:
                    action = legal[0]; forced += 1
                else:
                    request = request_for(observation, legal, rules, catalog, memory)
                    if len(json.dumps(request, ensure_ascii=False)) > max_prompt_chars:
                        status = 'context_limit'; break
                    action = None
                    for attempt in range(2):
                        if calls >= max_calls:break
                        calls += 1
                        path = directory/f'call_{calls:04d}'
                        dump(path.with_suffix('.request.json'), request)
                        response = invoke(adapter['command'], request, call_timeout, '/tmp')
                        dump(path.with_suffix('.response.json'), response)
                        if response.get('error') in ('rate_limit', 'authentication'):
                            status = 'provider_' + response['error']; break
                        try:
                            if not response.get('ok') or response.get('tool_events', 0):
                                raise ValueError('CLI did not produce an allowed successful response')
                            index, next_memory = parse_choice(response['text'], len(legal))
                            action = legal[index]; memory = next_memory; break
                        except (ValueError, KeyError, TypeError) as exc:
                            invalid += 1
                            request['format_correction'] = str(exc)
                    if action is None:
                        if status == 'running':status = 'agent_error'
                        break
                elapsed = time.monotonic()-before
                policy_seconds[actor] += elapsed
                if action not in legal:
                    raise ValueError('opponent returned an illegal action')
                event = dict(player=actor, observation=observation, legal_actions=legal,
                             action=copy_data(action), seconds=elapsed, memory=memory if actor==seat else None)
                if search and actor != seat:event['search'] = copy_data(opponent.last)
                env.step(action); decisions[actor] += 1
                trajectory.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                dump(directory/'progress.json', dict(status='running', calls=calls, invalid=invalid,
                    seat_decisions=decisions, wall_seconds=time.monotonic()-began,
                    search_seconds=opponent.spent if search else None, updated_unix=time.time()))
            if env.outcome()['terminated']:status = 'complete'
            if search and opponent.spent > entry['spec']['seconds']:status = 'budget_exceeded'
        except Exception as exc:
            status = 'runner_error'
            dump(directory/'error.json', dict(type=type(exc).__name__, message=str(exc)))
    result = dict(metadata, status=status, outcome=env.outcome(),
        score_breakdown=env.score_breakdown() if env.outcome()['terminated'] else None,
        calls=calls, invalid_responses=invalid, forced_agent_actions=forced,
        seat_decisions=decisions, policy_seconds=policy_seconds,
        search_seconds=opponent.spent if search else None,
        parent_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-began,
        ended_unix=time.time())
    if status == 'provider_rate_limit':
        saved = dict(snapshot=env.snapshot(), opponent=opponent,
            counters=(calls,invalid,forced,decisions,memory,policy_seconds),
            wall_seconds=result['wall_seconds'],cpu_seconds=result['parent_cpu_seconds'])
        blob=pickle.dumps(saved)
        temporary=resume_path.with_suffix('.part');temporary.write_bytes(blob);temporary.replace(resume_path)
        dump(directory/'resume.sha256.json',{'sha256':hashlib.sha256(blob).hexdigest()})
    dump(directory/'result.json', result)
    return result


def main():
    def terminated(signum, frame):
        raise RuntimeError('worker_terminated')
    signal.signal(signal.SIGTERM, terminated)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True);p.add_argument('--job', required=True)
    p.add_argument('--adapter', required=True, help='Private JSON containing command argv')
    p.add_argument('--rules', required=True);p.add_argument('--out', required=True)
    args=p.parse_args()
    manifest=verify_pool(args.pool)
    result=play_job(args.pool, manifest, json.loads(Path(args.job).read_text()),
        json.loads(Path(args.adapter).read_text()), Path(args.rules).read_text(), args.out)
    print(json.dumps({k:result[k] for k in ('status','outcome','calls','wall_seconds')},ensure_ascii=False))


if __name__ == '__main__':main()
