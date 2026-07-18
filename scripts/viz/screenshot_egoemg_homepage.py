"""
Render the EgoEMG academic project homepage and write PNG screenshots.

Why this is the working path on a headless server:
  * Default chromium waits for a GPU/compositor frame that never arrives and
    page.screenshot() hangs 60-90s.
  * --use-gl=swiftshader forces a software GL implementation that headless
    can use immediately.
  * set_content() with the CSS inlined and images inlined as data: URLs
    removes file:// I/O and network waits, so the page is laid out
    synchronously after domcontentloaded.

Usage:
    python scripts/viz/screenshot_egoemg_homepage.py
    python scripts/viz/screenshot_egoemg_homepage.py --output-dir /tmp/out
    python scripts/viz/screenshot_egoemg_homepage.py --doc-root <path>

By default the script reads from docs/egoemg_academic_homepage/ and writes
to docs/egoemg_academic_homepage/screenshots/.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# --- Defaults relative to the repo root -----------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_ROOT = REPO_ROOT / "docs" / "egoemg_academic_homepage"
DEFAULT_OUT_DIR = DEFAULT_DOC_ROOT / "screenshots"


# --- Section scroll offsets used for the desktop section captures ---------
DESKTOP_SECTIONS = [
    (0,    "01_hero"),
    (900,  "02_abstract_stats"),
    (1800, "03_capture_bench"),
    (2700, "04_results"),
    (3600, "05_gallery"),
    (4500, "06_resources_citation"),
]

VIEWPORTS = [
    ("desktop", {"width": 1440, "height": 900}),
    ("tablet",  {"width": 900,  "height": 1200}),
    ("mobile",  {"width": 390,  "height": 844}),
]

# Section scroll offsets. Override on the CLI by setting a custom list per
# project; the defaults match the hand-rolled EgoEMG v1 layout. The Horwitz
# template pass uses an empty list to skip section scrolls and only emit
# hero + full + mobile hero.
SECTION_OFFSETS: list[tuple[int, str]] = []

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--use-gl=swiftshader",
]


def build_inlined_html(doc_root: Path) -> str:
    """Inline stylesheets and image assets into a single HTML string.

    Supports two layouts:

    * Single-file: ``<link rel="stylesheet" href="styles.css" />`` next to
      ``styles.css`` (the original EgoEMG page).
    * External CSS: ``<link rel="stylesheet" href="static/css/...">`` with
      files in ``static/css/`` (the Horwitz template). All ``<link
      rel="stylesheet">`` to local files are inlined.
    """
    html = (doc_root / "index.html").read_text()

    top_level_css = doc_root / "styles.css"
    if top_level_css.exists():
        css = top_level_css.read_text()
        html = html.replace(
            '<link rel="stylesheet" href="styles.css" />',
            f"<style>{css}</style>",
        )

    def inline_link(match: re.Match) -> str:
        href = match.group(1)
        if href.startswith(("http://", "https://", "//")):
            return match.group(0)
        css_path = (doc_root / href).resolve()
        if not css_path.exists() or css_path.suffix != ".css":
            return match.group(0)
        return f"<style>{css_path.read_text()}</style>"

    html = re.sub(
        r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*/?>',
        inline_link,
        html,
    )

    mime_for = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "svg": "image/svg+xml",
        "gif": "image/gif",
        "webp": "image/webp",
    }

    def inline_asset(match: re.Match) -> str:
        p = doc_root / match.group(1)
        if not p.exists():
            return match.group(0)
        mime = mime_for.get(p.suffix.lstrip(".").lower())
        if mime is None:
            return match.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="(assets/[^"]+|static/images/[^"]+|static/pdfs/[^"]+)"', inline_asset, html)


def render(html: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=LAUNCH_ARGS)
        try:
            for label, viewport in VIEWPORTS:
                ctx = browser.new_context(
                    viewport=viewport, device_scale_factor=1
                )
                page = ctx.new_page()
                page.set_default_timeout(20000)
                page.set_content(html, wait_until="domcontentloaded")
                # Let layout settle; data: URLs decode synchronously.
                page.wait_for_timeout(500)

                if label == "desktop":
                    for scroll_y, name in DESKTOP_SECTIONS:
                        page.evaluate(f"window.scrollTo(0, {scroll_y})")
                        page.wait_for_timeout(150)
                        target = out_dir / f"desktop_{name}.png"
                        page.screenshot(
                            path=str(target), full_page=False, timeout=20000
                        )
                        print(f"  wrote {target.relative_to(REPO_ROOT)}")
                    full = out_dir / "desktop_full.png"
                    page.screenshot(
                        path=str(full), full_page=True, timeout=60000
                    )
                    print(f"  wrote {full.relative_to(REPO_ROOT)}")
                else:
                    target = out_dir / f"{label}_hero.png"
                    page.screenshot(
                        path=str(target), full_page=False, timeout=20000
                    )
                    print(f"  wrote {target.relative_to(REPO_ROOT)}")

                ctx.close()
        finally:
            browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the EgoEMG homepage to PNG screenshots."
    )
    parser.add_argument(
        "--doc-root",
        type=Path,
        default=DEFAULT_DOC_ROOT,
        help=f"Path to the homepage directory (default: {DEFAULT_DOC_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write PNGs (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc_root: Path = args.doc_root.resolve()
    out_dir: Path = args.output_dir.resolve()

    if not (doc_root / "index.html").exists():
        print(f"error: {doc_root / 'index.html'} not found", file=sys.stderr)
        return 1

    print(f"doc root: {doc_root}")
    print(f"out dir : {out_dir}")

    html = build_inlined_html(doc_root)
    render(html, out_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
