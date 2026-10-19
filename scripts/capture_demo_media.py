#!/usr/bin/env python
"""Capture the review UI as screenshots and a short screen recording.

The submission template asks for "a deployed link or a video demo". This produces the
second: it starts the server itself, drives the page with Playwright, writes PNGs for
the post and a WebM recording, then shuts the server down.

    python scripts/capture_demo_media.py

Outputs:
    reports/figures/ui_overview.png
    reports/figures/ui_review_field.png
    reports/demo_video/underwrite-demo.webm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

PORT = 8071


def wait_for_server(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise SystemExit(f"server did not come up at {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--video-dir", default="reports/demo_video")
    args = parser.parse_args()

    figures = Path(args.figures)
    video_dir = Path(args.video_dir)
    figures.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE / "src")
    env["UNDERWRITE_BATCH"] = str(HERE / "reports" / "demo_batch.json")

    log_path = video_dir / "server.log"
    log = log_path.open("wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "underwrite.app.server:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=HERE,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_server(f"http://127.0.0.1:{args.port}/api/health")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1360, "height": 900},
                record_video_dir=str(video_dir),
                record_video_size={"width": 1360, "height": 900},
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{args.port}/", wait_until="load")
            page.wait_for_selector(".doc", timeout=15000)
            page.wait_for_timeout(800)

            page.screenshot(path=str(figures / "ui_overview.png"), full_page=True)

            # Index 0 is already selected on load, so re-clicking it re-renders the same
            # view; start at 1 so the review screenshot shows a different document.
            rows = page.locator(".doc")
            total = min(8, rows.count())
            capture_review = True
            for index in range(1, total):
                rows.nth(index).click()
                page.wait_for_timeout(900)
                if capture_review and page.locator(".field.review").count() > 0:
                    page.screenshot(path=str(figures / "ui_review_field.png"), full_page=True)
                    capture_review = False
            page.wait_for_timeout(1200)

            video = page.video
            recorded = video.path() if video else None
            context.close()
            browser.close()

        if recorded:
            source = Path(recorded)
            target = video_dir / "underwrite-demo.webm"
            if source.is_file():
                source.replace(target)
                size_mb = target.stat().st_size / (1024 * 1024)
                print(f"video  -> {target}  ({size_mb:.1f} MB)")
            else:
                print(f"video reported at {source} but is missing")
        print(f"figure -> {figures / 'ui_overview.png'}")
        print(f"figure -> {figures / 'ui_review_field.png'}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
