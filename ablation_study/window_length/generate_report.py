#!/usr/bin/env python3
"""Generate a visually distinctive HTML report for the window-length ablation study.

Aesthetic: Dark industrial / scientific instrument — oscilloscope-inspired,
with neon cyan accents on deep charcoal backgrounds. Monospace data readouts.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_HTML = Path(__file__).resolve().parent / "report.html"

# Color palette — oscilloscope / lab instrument
BG_DARK = "#0a0e14"
BG_CARD = "#111820"
BG_CARD_HOVER = "#1a2230"
BORDER = "#1e2d3d"
CYAN = "#00e5ff"
CYAN_DIM = "#006680"
AMBER = "#ffab00"
RED = "#ff3d71"
GREEN = "#00e676"
TEXT = "#c8d6e5"
TEXT_MUTED = "#5c7080"
GRID_COLOR = "#1a2a3a"

PLOT_COLORS = ["#ff3d71", "#ff9100", "#ffea00", "#00e676", "#00e5ff", "#2979ff", "#d500f9", "#ff6090"]


def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_CARD,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT,
        "legend.facecolor": BG_CARD,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
        "font.family": "monospace",
        "font.size": 9,
    })


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_scatter_plot(trials: list[dict]) -> str:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))

    wls = [t["wl"] for t in trials]
    maes = [t["val_mae"] for t in trials]

    ax.scatter(wls, maes, s=100, c=CYAN, edgecolors=CYAN_DIM, linewidths=2, zorder=5, alpha=0.9)

    z = np.polyfit(wls, maes, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(wls) - 1000, max(wls) + 1000, 300)
    ax.plot(x_smooth, p(x_smooth), "--", color=AMBER, linewidth=1.5, alpha=0.7, label="Quadratic Fit")

    for t in trials:
        ax.annotate(
            f"{t['val_mae']:.4f}",
            (t["wl"], t["val_mae"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color=CYAN,
            fontweight="bold",
        )

    ax.set_xlabel("Window Length (samples @ 2kHz)", fontsize=10, labelpad=10)
    ax.set_ylabel("Val MAE (rad)", fontsize=10, labelpad=10)
    ax.set_title("Window Length vs. Val MAE", fontsize=13, fontweight="bold", color=CYAN, pad=15)
    ax.legend(fontsize=9, loc="upper right")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_timestep_plot(trial: dict, mae_array: np.ndarray) -> str:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    T = len(mae_array)
    x_pct = np.linspace(0, 100, T)

    ax.fill_between(x_pct, mae_array, alpha=0.15, color=CYAN)
    ax.plot(x_pct, mae_array, linewidth=0.8, color=CYAN, alpha=0.5)

    window_size = max(1, T // 40)
    smoothed = np.convolve(mae_array, np.ones(window_size) / window_size, mode="same")
    ax.plot(x_pct, smoothed, linewidth=2.2, color=CYAN, alpha=0.95)

    ax.axhline(mae_array.mean(), color=AMBER, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(98, mae_array.mean(), f" {mae_array.mean():.4f}", va="bottom", ha="right",
            fontsize=8, color=AMBER, fontweight="bold")

    min_idx = mae_array.argmin()
    max_idx = mae_array.argmax()
    ax.plot(min_idx / T * 100, mae_array[min_idx], "v", color=GREEN, markersize=8, zorder=10)
    ax.plot(max_idx / T * 100, mae_array[max_idx], "^", color=RED, markersize=8, zorder=10)

    ax.set_xlabel("Position in Window (%)", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    title = f"WL={trial['wl']:,}  |  MAE={trial['val_mae']:.4f}  |  T={trial['T_out']:,}  |  {trial['wl']/2000:.1f}s"
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_split_mae_plot() -> str:
    """Line chart with draggable HTML legend overlay.
    Returns (base64_img, legend_html) as a combined HTML snippet."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    split_results_path = RESULTS_DIR / "split_mae_results.json"
    if split_results_path.exists():
        with open(split_results_path) as f:
            split_results = json.load(f)
    else:
        split_results = []

    split_data = {"overall": [], "gesture": [], "user": [], "both": []}
    for r in split_results:
        wl = r["wl"]
        for key in split_data:
            if key in r:
                split_data[key].append((wl, r[key]))

    colors = {"overall": CYAN, "gesture": GREEN, "user": RED, "both": AMBER}
    markers = {"overall": "o", "gesture": "s", "user": "^", "both": "D"}

    for key in ["overall", "gesture", "user", "both"]:
        data = sorted(split_data[key])
        if not data:
            continue
        wls = [d[0] for d in data]
        maes = [d[1] for d in data]
        ax.plot(wls, maes, marker=markers[key], markersize=7, linewidth=2.2,
                color=colors[key], alpha=0.9)

    ax.set_xlabel("Window Length (samples @ 2kHz)", fontsize=10, labelpad=10)
    ax.set_ylabel("Test MAE (rad)", fontsize=10, labelpad=10)
    ax.set_title("Window Length vs. MAE by Generalization Split",
                 fontsize=13, fontweight="bold", color=CYAN, pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)

    legend_id = "splitLegend"
    html = f"""
    <div style="position:relative;">
        <img src="data:image/png;base64,{b64}" alt="Split MAE" style="width:100%;border-radius:4px;display:block;">
        <div id="{legend_id}" style="
            position:absolute; top:60px; right:30px;
            background:rgba(17,24,32,0.92); border:1px solid #1e2d3d;
            border-radius:6px; padding:12px 16px; cursor:move;
            font-family:'JetBrains Mono',monospace; font-size:12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            user-select:none; z-index:10;
        ">
            <div style="color:#5c7080;font-size:10px;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Legend (drag)</div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:14px;height:3px;background:{CYAN};margin-right:8px;vertical-align:middle;border-radius:2px;"></span><span style="color:{CYAN};">Overall (val MAE)</span></div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:14px;height:3px;background:{GREEN};margin-right:8px;vertical-align:middle;border-radius:2px;"></span><span style="color:{GREEN};">Gesture (seen user, unseen gesture)</span></div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:14px;height:3px;background:{RED};margin-right:8px;vertical-align:middle;border-radius:2px;"></span><span style="color:{RED};">User (unseen user)</span></div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:14px;height:3px;background:{AMBER};margin-right:8px;vertical-align:middle;border-radius:2px;"></span><span style="color:{AMBER};">Both (unseen user + gesture)</span></div>
        </div>
    </div>
    <script>
    (function() {{
        const el = document.getElementById('{legend_id}');
        let dragging = false, ox = 0, oy = 0;
        el.addEventListener('mousedown', function(e) {{
            dragging = true;
            ox = e.clientX - el.getBoundingClientRect().left;
            oy = e.clientY - el.getBoundingClientRect().top;
            el.style.transition = 'none';
            e.preventDefault();
        }});
        document.addEventListener('mousemove', function(e) {{
            if (!dragging) return;
            const p = el.parentElement.getBoundingClientRect();
            let x = e.clientX - p.left - ox;
            let y = e.clientY - p.top - oy;
            x = Math.max(0, Math.min(x, p.width - el.offsetWidth));
            y = Math.max(0, Math.min(y, p.height - el.offsetHeight));
            el.style.left = x + 'px';
            el.style.top = y + 'px';
            el.style.right = 'auto';
        }});
        document.addEventListener('mouseup', function() {{ dragging = false; }});
    }})();
    </script>"""
    return html


def make_overlay_plot(trials: list[dict], mae_arrays: dict[int, np.ndarray]) -> str:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    sorted_trials = sorted(trials, key=lambda t: t["wl"])
    for i, trial in enumerate(sorted_trials):
        wl = trial["wl"]
        if wl not in mae_arrays:
            continue
        arr = mae_arrays[wl]
        T = len(arr)
        x_pct = np.linspace(0, 100, T)
        window_size = max(1, T // 25)
        smoothed = np.convolve(arr, np.ones(window_size) / window_size, mode="same")
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.plot(x_pct, smoothed, linewidth=2, color=color, alpha=0.88,
                label=f"WL={wl:,} ({trial['val_mae']:.4f})")

    ax.set_xlabel("Position in Window (%)", fontsize=10, labelpad=10)
    ax.set_ylabel("MAE (rad)", fontsize=10, labelpad=10)
    ax.set_title("Per-Timestep MAE Comparison", fontsize=13, fontweight="bold", color=CYAN, pad=15)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.9)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_html(summary: dict, mae_arrays: dict[int, np.ndarray]) -> str:
    trials = summary["trials"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    scatter_b64 = make_scatter_plot(trials)
    split_mae_html = make_split_mae_plot()
    overlay_b64 = make_overlay_plot(trials, mae_arrays)

    timestep_plots = []
    for trial in sorted(trials, key=lambda t: t["wl"]):
        if trial["wl"] in mae_arrays:
            b64 = make_timestep_plot(trial, mae_arrays[trial["wl"]])
            timestep_plots.append((trial, b64))

    best = min(trials, key=lambda t: t["val_mae"])
    worst = max(trials, key=lambda t: t["val_mae"])
    improvement = (worst["val_mae"] - best["val_mae"]) / worst["val_mae"] * 100

    rows_html = ""
    for i, t in enumerate(sorted(trials, key=lambda x: x["wl"])):
        is_best = t["wl"] == best["wl"]
        row_class = "row-best" if is_best else ""
        rows_html += f"""
        <tr class="{row_class}">
            <td class="mono">{t['wl']:,}</td>
            <td class="mono">{t['wl']/2000:.2f}s</td>
            <td class="mono val-mae">{t['val_mae']:.4f}</td>
            <td class="mono">{t['overall_mae']:.4f}</td>
            <td class="mono">{t['min_mae']:.4f} <span class="pos">({t['min_mae_pos']*100:.0f}%)</span></td>
            <td class="mono">{t['max_mae']:.4f} <span class="pos">({t['max_mae_pos']*100:.0f}%)</span></td>
            <td class="mono">{t['max_mae'] - t['min_mae']:.4f}</td>
            <td class="mono">{t['T_out']:,}</td>
            <td class="study-tag">{t['study'].split('-')[-1]}</td>
        </tr>"""

    timestep_html = ""
    for trial, b64 in timestep_plots:
        timestep_html += f"""
        <div class="plot-panel">
            <img src="data:image/png;base64,{b64}" alt="WL={trial['wl']} 时间步分析">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>时间窗长度消融实验报告</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: {BG_DARK};
            --card: {BG_CARD};
            --card-hover: {BG_CARD_HOVER};
            --border: {BORDER};
            --cyan: {CYAN};
            --cyan-dim: {CYAN_DIM};
            --amber: {AMBER};
            --red: {RED};
            --green: {GREEN};
            --text: {TEXT};
            --text-muted: {TEXT_MUTED};
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Noto Sans SC', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
            min-height: 100vh;
        }}

        .noise-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            opacity: 0.03;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            z-index: 9999;
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 3rem 2rem;
        }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 4rem;
            position: relative;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 600px;
            height: 600px;
            background: radial-gradient(ellipse, rgba(0,229,255,0.04) 0%, transparent 70%);
            pointer-events: none;
        }}

        .header h1 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.4rem;
            font-weight: 700;
            color: var(--cyan);
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 40px rgba(0,229,255,0.3);
        }}

        .header .subtitle {{
            font-size: 1rem;
            color: var(--text-muted);
            font-weight: 300;
        }}

        .header .meta {{
            display: flex;
            justify-content: center;
            gap: 2.5rem;
            margin-top: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .header .meta span {{
            padding: 0.3rem 0.8rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--card);
        }}

        /* Sections */
        .section {{
            margin-bottom: 3rem;
        }}

        .section-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}

        .section-title::before {{
            content: '//';
            color: var(--cyan-dim);
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 3rem;
        }}

        .stat-cell {{
            background: var(--card);
            padding: 1.8rem 1.5rem;
            text-align: center;
            transition: background 0.2s;
        }}

        .stat-cell:hover {{
            background: var(--card-hover);
        }}

        .stat-cell .value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: var(--cyan);
            text-shadow: 0 0 20px rgba(0,229,255,0.2);
        }}

        .stat-cell .value.amber {{ color: var(--amber); text-shadow: 0 0 20px rgba(255,171,0,0.2); }}
        .stat-cell .value.green {{ color: var(--green); text-shadow: 0 0 20px rgba(0,230,118,0.2); }}
        .stat-cell .value.red {{ color: var(--red); text-shadow: 0 0 20px rgba(255,61,113,0.2); }}

        .stat-cell .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Plot panels */
        .plot-panel {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            transition: border-color 0.3s;
        }}

        .plot-panel:hover {{
            border-color: var(--cyan-dim);
        }}

        .plot-panel img {{
            width: 100%;
            border-radius: 4px;
            display: block;
        }}

        /* Table */
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}

        th {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--cyan-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 1rem 1rem;
            text-align: center;
            background: var(--card);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
        }}

        td {{
            padding: 0.8rem 1rem;
            text-align: center;
            border-bottom: 1px solid rgba(30,45,61,0.5);
            background: var(--bg);
        }}

        tr:hover td {{
            background: var(--card);
        }}

        tr.row-best td {{
            background: rgba(0,229,255,0.05);
            border-bottom-color: var(--cyan-dim);
        }}

        tr.row-best .val-mae {{
            color: var(--cyan);
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0,229,255,0.3);
        }}

        .mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .pos {{
            color: var(--text-muted);
            font-size: 0.75rem;
        }}

        .study-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--amber);
            background: rgba(255,171,0,0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            border: 1px solid rgba(255,171,0,0.2);
        }}

        /* Findings */
        .findings {{
            background: var(--card);
            border: 1px solid var(--border);
            border-left: 3px solid var(--cyan);
            border-radius: 0 8px 8px 0;
            padding: 2rem;
        }}

        .findings h3 {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan);
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
            letter-spacing: 1px;
        }}

        .findings ul {{
            list-style: none;
            padding: 0;
        }}

        .findings li {{
            padding: 0.6rem 0;
            padding-left: 1.5rem;
            position: relative;
            color: var(--text);
            font-size: 0.9rem;
            line-height: 1.8;
        }}

        .findings li::before {{
            content: '>';
            position: absolute;
            left: 0;
            color: var(--cyan-dim);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }}

        .findings li strong {{
            color: var(--cyan);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header h1 {{ font-size: 1.6rem; }}
            .header .meta {{ flex-wrap: wrap; gap: 0.5rem; }}
            .container {{ padding: 1.5rem 1rem; }}
        }}

        /* Animations */
        @keyframes glow-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}

        .stat-cell .value {{
            animation: glow-pulse 3s ease-in-out infinite;
        }}

        .stat-cell:nth-child(2) .value {{ animation-delay: 0.5s; }}
        .stat-cell:nth-child(3) .value {{ animation-delay: 1s; }}
        .stat-cell:nth-child(4) .value {{ animation-delay: 1.5s; }}
    </style>
</head>
<body>
<div class="noise-overlay"></div>
<div class="container">

    <div class="header">
        <h1>时间窗长度消融实验</h1>
        <p class="subtitle">EMGFormer Middle 模型 · EgoEMG 数据集 · 双向注意力机制</p>
        <div class="meta">
            <span>生成时间: {now}</span>
            <span>模型参数: 6.6M</span>
            <span>分析试验: {len(trials)} 组</span>
            <span>搜索范围: 7,494 – 29,000</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">核心指标</div>
        <div class="stats-grid">
            <div class="stat-cell">
                <div class="value">{best['val_mae']:.4f}</div>
                <div class="label">最优 MAE<br><span style="color:var(--cyan-dim)">WL={best['wl']:,}</span></div>
            </div>
            <div class="stat-cell">
                <div class="value red">{worst['val_mae']:.4f}</div>
                <div class="label">最差 MAE<br><span style="color:var(--cyan-dim)">WL={worst['wl']:,}</span></div>
            </div>
            <div class="stat-cell">
                <div class="value green">{improvement:.1f}%</div>
                <div class="label">相对提升</div>
            </div>
            <div class="stat-cell">
                <div class="value amber">{best['wl']/2000:.1f}s</div>
                <div class="label">最优窗口时长</div>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">时间窗长度 vs 验证集 MAE</div>
        <div class="plot-panel">
            <img src="data:image/png;base64,{scatter_b64}" alt="散点图">
        </div>
    </div>

    <div class="section">
        <div class="section-title">泛化能力分析: 各 Split MAE 随窗口长度变化</div>
        <div class="plot-panel">
            {split_mae_html}
        </div>
        <div class="findings" style="margin-top:1rem;">
            <h3>分析结论</h3>
            <ul>
                <li><strong>Gesture Split (已见用户):</strong> 窗口增大带来显著收益，MAE 从 0.226 (WL=7.5k) 降至 0.106 (WL=29k)，降幅 53%。模型能充分利用长时序上下文识别已见用户的手势模式。</li>
                <li><strong>User Split (未见用户):</strong> MAE 始终在 0.29–0.30 附近波动，窗口长度对跨用户泛化几乎无帮助。瓶颈在于用户间 EMG 信号分布差异（肌肉解剖、电极放置等个体因素）。</li>
                <li><strong>Both Split (未见用户+手势):</strong> 与 User Split 趋势一致，稳定在 0.30–0.31，进一步证实跨用户泛化是主要瓶颈。</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <div class="section-title">实验结果总表</div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>窗口长度</th>
                        <th>时长</th>
                        <th>Val MAE</th>
                        <th>整体 MAE</th>
                        <th>最小 MAE (位置)</th>
                        <th>最大 MAE (位置)</th>
                        <th>极差</th>
                        <th>输出步数</th>
                        <th>搜索轮次</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <div class="section-title">逐时间步 MAE 对比 (全部试验叠加)</div>
        <div class="plot-panel">
            <img src="data:image/png;base64,{overlay_b64}" alt="叠加对比图">
        </div>
    </div>

    <div class="section">
        <div class="section-title">各试验逐时间步分析</div>
        {timestep_html}
    </div>

    <div class="section">
        <div class="section-title">关键发现</div>
        <div class="findings">
            <h3>实验结论</h3>
            <ul>
                <li><strong>单调递减趋势:</strong> 更长的时间窗一致性地降低整体 MAE，在搜索范围内未观察到饱和现象。从 WL=7,494 到 WL=29,000，MAE 从 0.259 降至 0.193。</li>
                <li><strong>显著提升幅度:</strong> 最优配置相比最短窗口实现了 <strong>{improvement:.1f}%</strong> 的相对误差降低，证明双向注意力机制能有效利用更长的时序上下文。</li>
                <li><strong>U 形误差分布:</strong> 双向注意力在窗口中部（20%–60%）产生最低误差，两端因缺少单侧上下文而略高。这一对称分布验证了模型确实在利用双向信息。</li>
                <li><strong>泛化瓶颈在用户差异:</strong> 窗口增大仅对 Gesture Split（已见用户）有效，MAE 从 0.226 降至 0.106；而 User/Both Split（未见用户）始终在 0.29–0.31，不受窗口长度影响。跨用户泛化的瓶颈在于 EMG 信号的个体差异，非时序建模能力不足。</li>
                <li><strong>边缘效应可控:</strong> 虽然极差随窗口增大略有增长，但所有位置的绝对 MAE 均在改善，边缘退化不构成瓶颈。</li>
                <li><strong>计算代价权衡:</strong> 更长窗口需要更小的 batch size（WL=29,000 时 bs≈50 vs WL=7,494 时 bs≈519），单 epoch 训练时间相应增加。</li>
            </ul>
        </div>
    </div>

</div>
</body>
</html>"""
    return html


def main():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print("错误: summary.json 未找到。请先运行 run_analysis.py。")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    mae_arrays = {}
    for trial in summary["trials"]:
        npy_path = RESULTS_DIR / trial["npy_file"]
        if npy_path.exists():
            mae_arrays[trial["wl"]] = np.load(npy_path)

    print(f"已加载 {len(mae_arrays)} 个时间步数组")
    html = generate_html(summary, mae_arrays)

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"报告已生成: {OUTPUT_HTML}")
    print(f"文件大小: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
