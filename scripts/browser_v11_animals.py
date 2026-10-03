"""Directed real browser clicks; fixture setup uses only the existing save format."""
import argparse
from copy import deepcopy
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright
from boardbench.artifacts.store import write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.ui.server import make_server
from animal_browser_fixture import fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    scenario=fixture()
    write_json(out/'scenario.json',scenario)
    env=Harmonies()
    original=env.load_state(scenario['snapshot'])
    save_id='a'*32
    write_json(out/'saves'/f'{save_id}.json',dict(task={'id':'harmonies','version':'harmonies_solo_a_v1'},
               snapshot=scenario['snapshot'],seed=811,frames=[{'observation':original,'info':env.reset_info()}],info=env.reset_info()))
    server=make_server(config={'task':{'id':'harmonies','version':'harmonies_solo_a_v1'},
                              'save_directory':str(out/'saves'),'policy_root':str(out/'policies')},port=0)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    errors=[]
    names=dict(gray='山',blue='水',brown='树干',green='树冠',yellow='田',red='建筑')
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1280,'height':980})
            page.on('pageerror',lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('state !== null && !busy')
            page.locator('#save-id').fill(save_id)
            page.locator('#load').click()
            page.wait_for_function('!busy && state.seed === "811"')
            def click(action):
                previous=page.evaluate('state.observation.decision_index')
                if action['type']=='choose_bundle':
                    page.locator('#offers button').nth(action['bundle_id']).click()
                elif action['type']=='place_token':
                    page.locator('#pending button').filter(has_text=names[action['color']]).click()
                    page.locator(f'[data-cell="{action["cell_id"]}"]').click()
                elif action['type']=='take_card':
                    page.locator('#market .animal-card').nth(action['market_slot']).locator('button').click()
                elif action['type']=='place_animal':
                    cards=page.evaluate('state.observation.active_cards')
                    slot=next(i for i,c in enumerate(cards) if c['instance_id']==action['instance_id'])
                    page.locator('#active .animal-card').nth(slot).locator('button').click()
                    page.locator(f'[data-cell="{action["anchor_cell_id"]}"]').click()
                else:
                    page.locator('#turn-controls button').first.click()
                page.wait_for_function('n => !busy && state.observation.decision_index===n+1',arg=previous)
            for i,action in enumerate(scenario['actions'],1):
                expected=env.step(action).observation
                click(action)
                actual=page.evaluate('state.observation')
                assert actual==expected,(i,action)
                if i==scenario['milestones']['interleaved_take']:
                    assert sum(actual['pending_tokens'].values())==2 and actual['card_taken']
                if i==scenario['milestones']['slot_released']:
                    assert len(actual['active_cards'])==3 and actual['completed_cards'][0]['placed_count']==5
                    assert any(a['type']=='take_card' for a in actual['legal_actions'])
                if i in [scenario['milestones']['partial_save'],scenario['milestones']['completed_save']]:
                    page.locator('#save').click()
                    page.wait_for_function('!busy && state.save_id')
                    click(next(a for a in actual['legal_actions'] if a['type']=='place_token'))
                    page.locator('#load').click()
                    page.wait_for_function('i => !busy && state.observation.decision_index===i',arg=i)
                    assert page.evaluate('state.observation')==actual
                    page.screenshot(path=str(out/f'animals_{i}.png'),full_page=True)
            assert page.evaluate('state.observation')==scenario['final_observation']
            assert not page.locator('#error').inner_text()
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(out/'animals_mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            browser.close()
        assert not errors,errors
        write_json(out/'report.json',{'status':'passed','real_browser_clicks':len(scenario['actions']),
                   'checks':list(scenario['milestones']),'page_errors':errors,'mobile_no_overflow':True})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__=='__main__':
    main()
