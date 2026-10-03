"""Real browser acceptance for STOP, deadline restore and mobile layout."""
import argparse
from pathlib import Path
from threading import Thread
import time

from playwright.sync_api import sync_playwright

from boardbench.artifacts.store import read_json, write_json
from boardbench.ui.server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {'task': {'id': 'harmonies', 'version': 'harmonies_solo_a_v1'},
              'human_time_budget_seconds': 1200, 'save_directory': str(out / 'saves')}
    server = make_server(config=config, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 980})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('state !== null && !busy')
            assert '人工对局' in page.locator('#episode-clock').inner_text()
            page.locator('#offers button').first.click()
            page.wait_for_function('!busy && state.observation.decision_index===1')
            before = page.evaluate('JSON.stringify(state.observation)')
            page.locator('#stop-episode').click()
            page.wait_for_function('!busy && state.episode_finished')
            assert page.evaluate('JSON.stringify(state.observation)') == before
            assert page.evaluate('state.episode_end_reason') == 'stopped'
            assert page.locator('#bot-step').is_disabled()
            assert page.locator('#challenge').is_enabled()
            page.locator('#save').click()
            page.wait_for_function('!busy && state.save_id')
            saved = page.locator('#save-id').input_value()
            page.locator('#new').click()
            page.wait_for_function('!busy && !state.episode_finished')
            page.locator('#save-id').fill(saved)
            page.locator('#load').click()
            page.wait_for_function('!busy && state.episode_finished')
            assert page.evaluate('JSON.stringify(state.observation)') == before
            # Simulate an expired wall-clock save; loading must not reset its clock.
            save_path = out / 'saves' / (saved + '.json')
            value = read_json(save_path)
            value.update(episode_finished=False, end_reason=None, deadline_unix=time.time() - 1)
            write_json(save_path, value)
            page.locator('#load').click()
            page.wait_for_function('!busy && state.episode_end_reason==="deadline"')
            assert '时间截止' in page.locator('#notice').inner_text()
            page.screenshot(path=str(out / 'deadline_desktop.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(out / 'deadline_mobile.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            browser.close()
        assert not errors, errors
        write_json(out / 'report.json', {'status': 'passed', 'page_errors': errors,
                   'checks': ['legal_step', 'STOP_preserves_state', 'save_stop_restore', 'expired_save_no_reset', 'mobile_no_overflow']})
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    main()
