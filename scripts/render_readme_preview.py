#!/usr/bin/env python3
"""Render README.md into a GitHub-like preview (HTML + PDF + per-page PNGs).

Rendering backends, tried in order (first success wins):

  1. Chromium via Playwright, *headless* first (works on any machine with a
     display or working GL).
  2. Chromium via Playwright, *headed* on a throwaway Xvnc virtual display.
     Needed on truly headless GPU boxes (a real NVIDIA GPU works here for
     OpenGL, but Chromium is a compositing browser that wants a display surface;
     Xvnc provides one). This is why GL-related "special handling" is required.
  3. WeasyPrint + poppler (HTML+CSS->PDF->PNG, no browser / GPU / network), the
     universal offline fallback.

Why 1 & 2 over the system ``google-chrome --headless``: the bundled build here
hangs on network idle, and headless Chromium can't screenshot without a display
surface.

Font/SVG fixes baked into the HTML (see also *golden rules* below):
  * pin a glyph-complete font stack that is actually installed, else WeasyPrint
    drops digits / deg / plus-minus and falls back to serif;
  * pre-rasterize referenced SVGs to PNG, which WeasyPrint mis-scales inline.

Usage:
  python scripts/render_readme_preview.py [README.md] [--out /tmp/egoemg_preview]
  python scripts/render_readme_preview.py --backend chrome|weasyprint

Dependencies (active env / system):
  pip install playwright markdown      # Chromium (then `playwright install chromium`)
  pip install weasyprint markdown      # WeasyPrint fallback
  apt-get install poppler-utils librsvg2-bin  # PDF->PNG, SVG rasterizing
  Xvnc (or Xvfb)                       # only for headed-on-virtual-display path
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time


_VENDOR_CSS = "scripts/render_readme_assets/github-markdown.css"
_CSS_URL = ("https://raw.githubusercontent.com/sindresorhus/"
            "github-markdown-css/main/github-markdown.css")
_CACHE = pathlib.Path("/tmp/github-markdown.css")

# Glyph-complete stack actually installed on headless Linux. `!important`
# overrides github-markdown-css's -apple-system list.
_FONT_FIX = (
    "* { font-family:'Noto Sans','DejaVu Sans','Liberation Sans',"
    "sans-serif !important; }"
)


def _load_css(repo: pathlib.Path) -> str:
    vendored = repo / _VENDOR_CSS
    if vendored.exists():
        return vendored.read_text()
    if _CACHE.exists():
        return _CACHE.read_text()
    try:
        import urllib.request

        with urllib.request.urlopen(_CSS_URL, timeout=15) as r:
            css = r.read().decode()
        _CACHE.write_text(css)
        return css
    except Exception:  # noqa: BLE001 - degrade gracefully
        print("[warn] github-markdown.css unavailable; bare styling",
              file=sys.stderr)
        return ""


def _prepare_images(html: str, repo: pathlib.Path, out_dir: pathlib.Path) -> str:
    """Copy local images into out_dir; rasterize SVG -> PNG. Rewrite img src to
    the flattened basename so WeasyPrint resolves them from the HTML's dir."""
    img_re = re.compile(r'(<img[^>]*\bsrc=")(?P<src>[^"]+)(?P<rest>"[^>]*>)')

    def repl(m: re.Match[str]) -> str:
        src = m.group("src").split("?")[0].split("#")[0]
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (repo / src).resolve()
        if not p.exists():
            return m.group(0)
        if p.suffix.lower() == ".svg":
            out = out_dir / (p.stem + ".png")
            try:
                subprocess.run(["rsvg-convert", "-w", "2400", "-o", str(out),
                                str(p)], check=True, capture_output=True)
            except FileNotFoundError:
                print(f"[warn] rsvg-convert missing; leaving SVG inline: {src}",
                      file=sys.stderr)
                return m.group(0)
            target = out.name
        else:
            target = p.name
            shutil.copy2(p, out_dir / target)
        return f'{m.group(1)}{target}{m.group("rest")}'

    return img_re.sub(repl, html)


_EXTENSIONS = ["tables", "fenced_code", "toc", "attr_list", "sane_lists"]
_DETAILS_RE = re.compile(r"<details>(.*?)</details>", re.S)
_SUMMARY_RE = re.compile(r"(\s*<summary>.*?</summary>)(.*)", re.S)


def _render_details(details: str) -> str:
    """Render markdown *inside* a `<details>` block. GitHub (cmark-gfm) parses
    markdown within `<details>`, but python-markdown leaves it as literal text,
    so we pre-render the inner body before the single outer pass."""
    import markdown

    m = _SUMMARY_RE.match(details)
    if not m:
        inner = markdown.markdown(details, extensions=_EXTENSIONS,
                                  output_format="html5")
        return f"<details>{inner}</details>"
    summary, body = m.group(1), m.group(2)
    inner = markdown.markdown(body, extensions=_EXTENSIONS,
                              output_format="html5")
    return f"<details>{summary}{inner}</details>"


def _md_to_html(md_text: str, css: str, repo: pathlib.Path,
                out_dir: pathlib.Path) -> str:
    import markdown

    md_text = _DETAILS_RE.sub(lambda m: _render_details(m.group(1)), md_text)
    body = markdown.markdown(
        md_text,
        extensions=_EXTENSIONS,
        output_format="html5",
    )
    body = _prepare_images(body, repo, out_dir)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{css}</style>
<style>
  body {{ background:#fff; margin:0; padding:24px 0; }}
  .markdown-body {{ max-width:900px; margin:0 auto; padding:0 24px; }}
  .markdown-body img {{ display:block; max-width:100%; height:auto; margin:0 auto; }}
  {_FONT_FIX}
</style>
</head><body><article class="markdown-body">{body}</article></body></html>"""


def _find_chrome() -> str | None:
    globs = [
        pathlib.Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux64/chrome",
        pathlib.Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
    ]
    for g in globs:
        hits = glob.glob(str(g))
        if hits:
            return hits[-1]
    return shutil.which("chromium") or shutil.which("google-chrome")


def _offline(html_file: pathlib.Path, page, headless: bool) -> None:
    """Block http(s) so captures are deterministic offline; file:// still loads.
    External shields.io badges then show as broken placeholder images."""
    page.route("**/*",
               lambda r: (r.abort() if r.request.url
                          .startswith(("http://", "https://"))
                          else r.continue_()))
    page.goto(html_file.as_uri(), wait_until="domcontentloaded")


def _shot(page, out: pathlib.Path) -> None:
    page.wait_for_timeout(1200)
    page.screenshot(path=str(out), full_page=True, timeout=30000)


def _headed_on_xvnc(html_file: pathlib.Path, out: pathlib.Path) -> bool:
    """Launch Chromium *headed* on a throwaway Xvnc display. On a headless box
    with a real GPU this is the reliable way to get pixels; headless Chromium
    can't composite without a display surface."""
    xserver = shutil.which("Xvnc") or "/usr/bin/Xvnc"
    chrome = _find_chrome()
    if not chrome or not os.path.exists(xserver):
        return False
    try:
        import playwright  # noqa: F401 - ensures the driver is importable
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    xvnc = subprocess.Popen(
        [xserver, ":99", "-geometry", "1920x1080", "-depth", "24",
         "-ac", "-localhost"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            if glob.glob("/tmp/.X11-unix/X99"):
                break
            time.sleep(0.5)
        env = {**os.environ, "DISPLAY": ":99"}
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chrome, headless=False,
                args=["--no-sandbox", "--disable-dev-shm-usage"], env=env,
            )
            page = browser.new_page(viewport={"width": 900, "height": 1200},
                                    device_scale_factor=2)
            _offline(html_file, page, headless=False)
            _shot(page, out)
            browser.close()
        return out.exists()
    except Exception:  # noqa: BLE001
        return False
    finally:
        xvnc.terminate()


def _chrome_capture(html_file: pathlib.Path,
                    out_dir: pathlib.Path) -> pathlib.Path | None:
    """High-fidelity full-page screenshot via Chromium. Returns the PNG or None.
    Tries headless first, then headed-on-Xvnc for GPU-less-display boxes."""
    chrome = _find_chrome()
    if not chrome:
        return None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    out = out_dir / "preview.png"
    # Attempt 1: headless (normal machines).
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chrome, headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--disable-background-networking"])
            page = browser.new_page(viewport={"width": 900, "height": 1200},
                                    device_scale_factor=2)
            _offline(html_file, page, headless=True)
            _shot(page, out)
            browser.close()
        if out.exists():
            return out
    except Exception:  # noqa: BLE001
        pass

    # Attempt 2: headed on a virtual display (headless GPU boxes).
    if _headed_on_xvnc(html_file, out):
        return out

    return None


def _to_pngs(pdf: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    try:
        subprocess.run(["pdftoppm", "-png", "-r", "70", str(pdf),
                        str(out_dir / "page")], check=True, capture_output=True)
    except FileNotFoundError:
        print("[warn] pdftoppm missing; skipping PNG output", file=sys.stderr)
        return []
    return sorted(out_dir.glob("page-*.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("md", nargs="?", default="README.md")
    parser.add_argument("--out", default="/tmp/egoemg_preview")
    parser.add_argument("--backend", choices=["auto", "chrome", "weasyprint"],
                        default="auto")
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    md_path = repo / args.md
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not md_path.exists():
        print(f"error: {md_path} not found", file=sys.stderr)
        return 1

    css = _load_css(repo)
    html = _md_to_html(md_path.read_text(), css, repo, out_dir)
    html_file = out_dir / "preview.html"
    html_file.write_text(html)
    print(f"HTML : {html_file}")

    # Preferred path: Chromium (headless, else headed-on-Xvnc).
    if args.backend in ("auto", "chrome"):
        shot = _chrome_capture(html_file, out_dir)
        if shot is not None:
            print(f"PNG  : {shot}  (Chromium, full page)")
            return 0

    # Offline fallback: WeasyPrint + poppler.
    try:
        import weasyprint

        weasyprint.HTML(filename=str(html_file)).write_pdf(str(out_dir / "preview.pdf"))
    except ImportError:
        print("[warn] weasyprint also unavailable; HTML written only "
              "(`pip install weasyprint`).", file=sys.stderr)
        return 0

    pngs = _to_pngs(out_dir / "preview.pdf", out_dir)
    print(f"PDF  : {out_dir / 'preview.pdf'}")
    print(f"PNGs : {len(pngs)} page(s) at {out_dir}  (WeasyPrint fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
