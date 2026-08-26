#!/usr/bin/env python3
"""One-command browser screenshots on this headless GPU box.

Chromium's compositor never produces a frame in headless mode here (real
NVIDIA GPU, no display surface -- see scripts/render_readme_assets/
TOOLCHAIN.md), so this script drives HEADED Playwright Chromium on an Xvnc
virtual display instead.

Usage:
  python scripts/dev/screenshot_url.py URL [URL ...] [--out DIR_OR_FILE]
      [--width 1440] [--height 900] [--full-page] [--wait-ms 4000]
      [--block-media] [--display :99]

- Reuses an already-running Xvnc on the given display and only starts a
  throwaway one (cleaned up on exit) when none exists.
- If the current interpreter lacks playwright, re-execs the first conda env
  python that has it (searches ~/miniconda3/envs/*).
- --block-media aborts video/audio requests so pages with remote videos
  reach a quiescent state quickly (use when you only need layout).
"""

from __future__ import annotations

import argparse
import atexit
import glob
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PLAYWRIGHT_ENV_GLOB = os.path.expanduser("~/miniconda3/envs/*/bin/python")


def log(msg: str) -> None:
    print(f"[screenshot] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- display ---

def display_in_use(name: str) -> bool:
    n = name.lstrip(":").split(".")[0]
    if Path(f"/tmp/.X11-unix/X{n}").exists():
        return True
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", 6000 + int(n))) == 0


def ensure_display(name: str, width: int, height: int) -> subprocess.Popen | None:
    """Return the Xvnc process if we started one, else None (reused)."""
    if display_in_use(name):
        log(f"reusing existing X server on {name}")
        return None
    geometry = f"{max(width, 1024)}x{max(height, 768)}x24"
    proc = subprocess.Popen(
        ["Xvnc", name, "-geometry", geometry, "-depth", "24", "-ac", "-localhost"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if display_in_use(name):
            log(f"started Xvnc on {name} (geometry {geometry})")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"Xvnc exited with code {proc.returncode}")
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("Xvnc did not come up within 5s")


# --------------------------------------------------------------- playwright --

def reexec_with_playwright() -> None:
    """Re-exec under the first conda env python that imports playwright."""
    if os.environ.get("_EGOEMG_SHOT_REEXEC"):
        raise RuntimeError("playwright not importable in any searched env")
    for py in sorted(glob.glob(PLAYWRIGHT_ENV_GLOB)):
        probe = subprocess.run(
            [py, "-c", "import playwright"], capture_output=True)
        if probe.returncode == 0:
            log(f"re-execing with {py}")
            os.environ["_EGOEMG_SHOT_REEXEC"] = "1"
            os.execve(py, [py, *sys.argv], os.environ)
    raise RuntimeError(
        f"no python with playwright found under {PLAYWRIGHT_ENV_GLOB}; "
        "pip install playwright into some env first")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", default="/tmp/screenshot",
                    help="output png path (single URL) or directory (multiple)")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--full-page", action="store_true")
    ap.add_argument("--wait-ms", type=int, default=4000,
                    help="fixed wait after domcontentloaded")
    ap.add_argument("--block-media", action="store_true",
                    help="abort video/audio requests (faster, layout-only)")
    ap.add_argument("--display", default=":99")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        reexec_with_playwright()
        return 2  # unreachable

    xvnc = ensure_display(args.display, args.width, args.height)
    if xvnc is not None:
        atexit.register(lambda: (xvnc.terminate(), xvnc.wait()))

    os.environ.setdefault("DISPLAY", args.display)

    out = Path(args.out)
    multiple = len(args.urls) > 1
    if multiple:
        out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # headed on Xvnc: headless never composites here
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height})
            if args.block_media:
                ctx.route(
                    "**/*",
                    lambda r: r.abort()
                    if r.request.resource_type in ("media",)
                    else r.continue_())
            page = ctx.new_page()
            for url in args.urls:
                log(f"capturing {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(args.wait_ms)
                dest = (out / _slug(url)) if multiple else (out.with_suffix(".png"))
                dest.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(
                    path=str(dest), full_page=args.full_page, timeout=60000)
                log(f"wrote {dest}")
        finally:
            browser.close()
    return 0


def _slug(url: str) -> str:
    s = url.split("://", 1)[-1].replace(":", "_").replace("/", "_")
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120] + ".png"


if __name__ == "__main__":
    sys.exit(main())
