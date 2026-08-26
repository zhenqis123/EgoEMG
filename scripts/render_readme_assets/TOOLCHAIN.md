# README preview toolchain

Renders `README.md` into a GitHub-like preview so edits can be eyeballed.

```bash
python scripts/render_readme_preview.py            # → /tmp/egoemg_preview/{preview.html,preview.pdf,page-*.png}
python scripts/render_readme_preview.py --out out/ # custom output dir
python scripts/render_readme_preview.py --backend weasyprint  # skip Chromium
```

## How it works

`markdown (python)` → styled HTML → renderer → PDF/PNG. Two backends:

| backend | fidelity | here | deps |
|---|---|---|---|
| Chromium (Playwright) | pixel-perfect (real CSS/fonts/SVG/emoji) | ✅ headless, else Xvnc | `pip install playwright && playwright install chromium` + `Xvnc` |
| WeasyPrint (default fallback) | good enough for layout/typography review | ✅ works offline | `pip install weasyprint markdown` + `apt-get install poppler-utils librsvg2-bin` |

The script tries Chromium first, then falls back to WeasyPrint automatically.
Chromium itself tries **headless** first (the fast path on any machine with a
display or working GL), and if that can't capture a frame it relaunches
**headed on a throwaway Xvnc virtual display** — the special handling a
display-less GPU box needs. Clean-up is automatic.

## Why WeasyPrint needs three explicit fixes

On this (and many) headless Linux machines, a naive markdown→HTML→WeasyPrint
pipeline silently corrupts output:

1. **Font fallback drops glyphs.** `github-markdown-css` names fonts absent
   here (`-apple-system`, `Helvetica`, `Arial`), so text falls back unpredictably
   and **digits / `±` / `°` / `Δ` get dropped** — a README full of numbers looks
   blank. Fix (in `render_readme_preview.py`): pin `* { font-family:'Noto Sans',
   'DejaVu Sans', ... , sans-serif !important }` (a stack that is actually
   installed, glyph-complete).
2. **SVG-in-`<img>` is mis-scaled by WeasyPrint** → the hero banner clips. Fix:
   pre-rasterize every referenced `.svg` to PNG (`rsvg-convert`) and rewrite the
   `src` to the PNG.
3. **External assets degrade offline.** `shields.io` badges stay uncolored and
   `github-markdown-css` must be vendored locally (this file's directory) so the
   pipeline is offline-repeatable.

`dataset_stats.svg` was also updated to use a cross-platform font stack (the old
`Arial, Helvetica, sans-serif` fell back to serif on Linux); its orphaned
sibling `dataset_stats.png` was deleted — the preview script rasterizes the SVG
at render time, so no checked-in raster is needed.

## The Chromium screenshot "unlock" on a headless GPU box

Headless Chromium (and the bundled `google-chrome`) loads pages — DOM, fonts,
layout all fine — but the compositor never produces a frame without a display
surface, so screenshots hang at "taking page screenshot / fonts loaded", even
with `--use-angle=swiftshader --enable-unsafe-swiftshader`. `--disable-gpu`
doesn't help either; Chromium is a compositing browser.

The fix (and the "special handling" a headless box with a real NVIDIA GPU
actually needs): give Chromium a display surface via **Xvnc**, then run it
**headed**:

```bash
Xvnc :99 -geometry 1920x1080 -depth 24 -ac -localhost &   # virtual display
DISPLAY=:99 python -c "..."                                 # headed Chromium screenshot
```

On this box GL *does* work (RTX 4090, driver 595.91.07) — that's why the GPU
headaches kept pointing elsewhere. The unlock is just the virtual *display*:
with `DISPLAY=:99`, headed Chromium composites and screenshots the whole page.
The script automates this: try headless, else spin up Xvnc + headed, tear down
after.

> Two traps when scripting this by hand: don't kill the Xvnc with a
> `pkill -f "Xvnc :99"` *inside the command that launches it* (the pattern
> matches that shell's own command line, so it self-kills), and pass
> `-localhost` as a bare flag (TigerVNC parses `-localhost no` as an unknown
> option).

## Markdown inside `<details>` (collapsible sections)

GitHub (cmark-gfm) parses markdown — tables, bold, fenced code — inside a
`<details>` block; `python-markdown` does **not** (it leaves the inner text as
literals). Fix (in `render_readme_preview.py`): pre-render each `<details>`
block's inner body with `markdown.markdown(...)` (splitting off the
`<summary>`) before the single outer pass. The README source stays GitHub-
correct; the renderer reproduces the behavior. (`md_in_html` does *not* handle
`<details>`, so don't rely on it.)

## Golden rules (from debugging this)

- **Smoke-test the renderer on a 1-line page before a real document.**
- Reach for a **headless-DOM renderer** (WeasyPrint / wkhtmltopdf) instead of a
  GUI-browser screenshot when there's no display/GPU/network.
- When a render looks broken, try to separate "renderer bug" from "document
  bug" before changing the README (e.g. WeasyPrint dropped the table digits —
  the source was fine; tables inside `<details>` looked literal — the source
  was fine, the renderer needed a fix).
