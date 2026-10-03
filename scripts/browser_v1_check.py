"""Real browser acceptance: manual actions, hint, takeover, save and challenge."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

from playwright.sync_api import sync_playwright
from boardbench.artifacts.store import write_json


@contextmanager
def server(python, game):
    process = subprocess.Popen([python, '-m', 'boardbench', 'play', '--config', f'configs/{game}_play.json', '--port', '0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    try:
        line = process.stdout.readline()
        if not line.startswith('BoardBench: http'):
            raise RuntimeError(line+process.stderr.read())
        yield line.strip().split(' ', 1)[1]
    finally:
        process.terminate()
        process.wait(timeout=15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--server-python', default='.venv-train/bin/python')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    errors, reports = [], {}
    names = dict(gray='山', blue='水', brown='树干', green='树冠', yellow='田', red='建筑')
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 980})
        page.on('pageerror', lambda e: errors.append(str(e)))
        for game in ('micro_tiles', 'take_it_easy', 'harmonies'):
            with server(args.server_python, game) as address:
                page.goto(address)
                page.wait_for_function('state !== null && !busy')
                policy_ids = page.locator('#policy option').evaluate_all('(nodes)=>nodes.map(n=>n.value)')
                trained = next((x for x in policy_ids if x.startswith('rl_trained')), None)
                page.locator('#policy').select_option('heuristic')
                before = page.evaluate('JSON.stringify(state.observation)')
                page.locator('#hint').click()
                page.wait_for_function('state.hint !== null && !busy')
                assert before == page.evaluate('JSON.stringify(state.observation)')
                page.locator('#reject').click()
                assert before == page.evaluate('JSON.stringify(state.observation)')
                # New seed is a string: JS must not silently round 64-bit integers.
                page.locator('#seed').fill('20261003')
                page.locator('#new').click()
                page.wait_for_function('state.seed === "20261003" && !busy')
                manual = 0
                def click_action(action):
                    previous = page.evaluate('state.observation.decision_index')
                    if game == 'micro_tiles':
                        page.locator('#offers button').nth(action['offer_index']).click()
                        page.locator(f'[data-cell="{action["cell_id"]}"]').click()
                    elif game == 'take_it_easy':
                        page.locator(f'[data-cell="{action["cell_id"]}"]').click()
                    elif action['type'] == 'choose_bundle':
                        page.locator('#offers button').nth(action['bundle_id']).click()
                    elif action['type'] == 'place_token':
                        page.locator('#pending button').filter(has_text=names[action['color']]).click()
                        page.locator(f'[data-cell="{action["cell_id"]}"]').click()
                    elif action['type'] == 'take_card':
                        page.locator('#market .animal-card').nth(action['market_slot']).locator('button').click()
                    elif action['type'] == 'place_animal':
                        cards = page.evaluate('state.observation.active_cards')
                        slot = next(i for i,c in enumerate(cards) if c['instance_id'] == action['instance_id'])
                        page.locator('#active .animal-card').nth(slot).locator('button').click()
                        page.locator(f'[data-cell="{action["anchor_cell_id"]}"]').click()
                    else:
                        page.locator('#turn-controls button').first.click()
                    page.wait_for_function('n => state.observation.decision_index === n+1 && !busy', arg=previous)
                while not page.evaluate('state.observation.terminated'):
                    action = page.evaluate('state.observation.legal_actions[0]')
                    click_action(action)
                    manual += 1
                    if manual == 2:
                        before = page.evaluate('JSON.stringify(state.observation)')
                        page.locator('#save').click()
                        page.wait_for_function('state.save_id && !busy')
                        save_id = page.locator('#save-id').input_value()
                        assert len(save_id) == 32
                        click_action(page.evaluate('state.observation.legal_actions[0]'))
                        page.locator('#load').click()
                        page.wait_for_function('!busy && state.observation.decision_index === 2')
                        assert before == page.evaluate('JSON.stringify(state.observation)')
                    assert manual < 256
                manual_score = page.evaluate('state.observation.score')
                page.screenshot(path=str(out/f'{game}_manual.png'), full_page=True)
                page.locator('#challenge').click()
                page.wait_for_function('state.challenge !== null && !busy', timeout=60000)
                page.locator('#challenge-position').fill('1')
                page.locator('#challenge-position').dispatch_event('input')
                assert page.evaluate('localReplay') == 1
                page.locator('#replay-close').click()
                page.locator('#new').click()
                page.wait_for_function('!busy && state.observation.decision_index === 0')
                if trained:
                    page.locator('#policy').select_option(trained)
                page.locator('#auto').click()
                page.wait_for_function('state.observation.decision_index >= 2', timeout=30000)
                page.locator('#pause').click()
                page.wait_for_function('!busy && !auto')
                paused = page.evaluate('state.observation.decision_index')
                time.sleep(.25)
                assert page.evaluate('state.observation.decision_index') == paused
                page.locator('#auto').click()
                page.wait_for_function('state.observation.terminated && !auto && !busy', timeout=90000)
                page.screenshot(path=str(out/f'{game}_policy.png'), full_page=True)
                page.set_viewport_size({'width': 390, 'height': 844})
                page.screenshot(path=str(out/f'{game}_mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                page.set_viewport_size({'width': 1280, 'height': 980})
                reports[game] = {'manual_score': manual_score, 'manual_actions': manual,
                                 'policy_score': page.evaluate('state.observation.score'),
                                 'trained_policy': trained, 'policy_choices': policy_ids,
                                 'checks': ['manual full game', 'hint does not mutate', 'reject hint',
                                            'midturn save/resume', 'challenge and replay', 'auto takeover',
                                            'pause and resume', 'mobile no overflow']}
                assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()
                write_json(out/'browser_check.json', {'status': 'in_progress', 'games': reports, 'page_errors': errors})
        browser.close()
    assert not errors, errors
    write_json(out/'browser_check.json', {'status': 'passed', 'games': reports, 'page_errors': errors})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
