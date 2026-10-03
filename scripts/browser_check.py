"""Actual headless browser interaction using the shipped CLI, with screenshots."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from playwright.sync_api import sync_playwright

from boardbench.artifacts.store import write_json


@contextmanager
def server(command):
    process = subprocess.Popen([sys.executable, "-m", "boardbench", *command, "--port", "0"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline()
        if not line.startswith("BoardBench: http"):
            raise RuntimeError(line + process.stderr.read())
        yield line.strip().split(" ", 1)[1]
    finally:
        process.terminate()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        with server(["play", "--config", "configs/risk_play.json"]) as address:
            page.goto(address)
            page.wait_for_function("document.querySelector('#mode').textContent.includes('第')")
            page.locator("#draw").click()
            page.wait_for_function("document.querySelector('#steps').textContent === '1'")
            page.wait_for_function("document.querySelector('#new').disabled === false")
            if page.locator("#bank").is_enabled():
                page.locator("#bank").click()
            page.wait_for_function("document.querySelector('#score').textContent !== '—'")
            assert page.locator("#draw").is_disabled()
            page.screenshot(path=str(out / "play.png"), full_page=True)
            final_score = page.locator("#score").inner_text()
            page.locator("#new").click()
            page.wait_for_function("document.querySelector('#steps').textContent === '0' && !document.querySelector('#draw').disabled")
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(out / "play_mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.set_viewport_size({"width": 1100, "height": 900})
        with server(["replay", "--trajectory", args.trajectory]) as address:
            page.goto(address)
            page.wait_for_function("document.querySelector('#mode').textContent === '轨迹回放'")
            page.locator("#next").click()
            page.wait_for_function("document.querySelector('#position').textContent.startsWith('1 /')")
            page.locator("#prev").click()
            page.wait_for_function("document.querySelector('#position').textContent.startsWith('0 /')")
            page.locator("#episode").select_option("2")
            page.wait_for_function("document.querySelector('#episode').value === '2' && !document.querySelector('#next').disabled")
            while page.locator("#next").is_enabled():
                current = page.locator("#position").inner_text()
                page.locator("#next").click()
                page.wait_for_function("old => document.querySelector('#position').textContent !== old", arg=current)
            page.screenshot(path=str(out / "replay.png"), full_page=True)
            replay_score = page.locator("#score").inner_text()
        browser.close()
    assert not errors, errors
    write_json(out / "browser_check.json", {
        "status": "passed", "browser": "Chromium via Playwright", "play_score": final_score,
        "replay_episode": 2, "replay_score": replay_score, "page_errors": errors,
        "checks": ["draw", "terminal score", "disabled terminal actions", "new game",
                   "mobile layout", "replay next", "replay previous", "episode selection", "terminal replay"],
        "screenshots": ["play.png", "play_mobile.png", "replay.png"]})
    print(f"Browser play/replay passed: {out}")


if __name__ == "__main__":
    main()
