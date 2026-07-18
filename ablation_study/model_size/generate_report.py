#!/usr/bin/env python3
"""Generate HTML report for emg2pose_v3 model size scaling ablation."""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "model_scaling_results.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "report.html")

with open(DATA_FILE) as f:
    data = json.load(f)

models = data["models"]
model_order = ["middle", "large", "xlarge", "xxlarge", "huge", "huge_last"]

# Extract summary data
display_names = {
    "middle": "Middle",
    "large": "Large",
    "xlarge": "XLarge",
    "xxlarge": "XXLarge",
    "huge": "Huge",
    "huge_last": "Huge (last)",
}

summary = []
for name in model_order:
    m = models[name]
    tr = m["test_results"]["stage"]
    ur = m["test_results"]["user"]
    summary.append({
        "key": name,
        "name": display_names.get(name, name.capitalize()),
        "params_M": m["total_params_M"],
        "decoder_params_M": m["decoder_params_M"],
        "model_dim": m["model_dim"],
        "num_heads": m["num_heads"],
        "num_layers": m["num_layers"],
        "ffn_dim": m["ffn_dim"],
        "best_val_mae": m["best_val_mae"],
        "best_epoch": m["best_epoch"],
        "epochs_trained": m["total_epochs_trained"],
        "stage_mae": tr["test_mae"],
        "user_mae": ur["test_mae"],
        "stage_ft_dist": tr["test_landmark/fingertip"],
        "user_ft_dist": ur["test_landmark/fingertip"],
        "stage_landmark": tr["test_landmark/all"],
        "stage_vel": tr["test_derivatives/vel"],
        "stage_acc": tr["test_derivatives/acc"],
        "stage_jerk": tr["test_derivatives/jerk"],
        "stage_loss": tr["test_loss"],
        "stage_proximal": tr["test_mae_per_pd/proximal"],
        "stage_mid": tr["test_mae_per_pd/mid"],
        "stage_distal": tr["test_mae_per_pd/distal"],
        "stage_thumb": tr["test_mae_per_finger/thumb"],
        "stage_index": tr["test_mae_per_finger/index"],
        "stage_middle_finger": tr["test_mae_per_finger/middle"],
        "stage_ring": tr["test_mae_per_finger/ring"],
        "stage_pinky": tr["test_mae_per_finger/pinky"],
    })

# Compute improvements relative to middle
for s in summary:
    middle = summary[0]
    s["improvement_over_middle_pct"] = (
        (middle["stage_mae"] - s["stage_mae"]) / middle["stage_mae"] * 100
    )
    s["ft_improvement_pct"] = (
        (middle["stage_ft_dist"] - s["stage_ft_dist"]) / middle["stage_ft_dist"] * 100
    )
    s["generalization_gap"] = s["user_mae"] - s["stage_mae"]
    s["generalization_ratio"] = s["stage_mae"] / s["user_mae"] * 100

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>emg2pose_v3 Model Size Scaling Ablation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e4e4e7; --text2: #9ca3af; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --blue: #3b82f6;
    --purple: #a855f7; --teal: #14b8a6;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:24px; line-height:1.6; }}
  .container {{ max-width:1400px; margin:0 auto; }}
  h1 {{ font-size:28px; font-weight:700; margin-bottom:8px; background:linear-gradient(135deg,var(--accent),var(--purple)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h2 {{ font-size:20px; font-weight:600; margin:32px 0 16px; color:var(--text); border-left:3px solid var(--accent); padding-left:12px; }}
  h3 {{ font-size:16px; font-weight:600; margin:20px 0 12px; color:var(--text2); }}
  .subtitle {{ color:var(--text2); font-size:14px; margin-bottom:32px; }}
  .meta {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }}
  .meta-tag {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:8px 14px; font-size:13px; color:var(--text2); }}
  .meta-tag strong {{ color:var(--text); }}

  /* Cards */
  .grid {{ display:grid; gap:16px; }}
  .grid-5 {{ grid-template-columns: repeat(5, 1fr); }}
  .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  .card-label {{ font-size:12px; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }}
  .card-value {{ font-size:24px; font-weight:700; }}
  .card-sub {{ font-size:12px; color:var(--text2); margin-top:4px; }}
  .card.highlight {{ border-color:var(--accent); background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(168,85,247,0.05)); }}
  .improve {{ color:var(--green); }}
  .worse {{ color:var(--red); }}
  .best {{ color:var(--accent); }}

  /* Tables */
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:rgba(99,102,241,0.1); color:var(--accent); font-weight:600; text-align:left; padding:10px 12px; border-bottom:2px solid var(--border); white-space:nowrap; }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
  tr:hover {{ background:rgba(255,255,255,0.02); }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .best-cell {{ color:var(--accent); font-weight:600; }}
  .bar-cell {{ position:relative; }}
  .bar-bg {{ position:absolute; left:0; top:50%; transform:translateY(-50%); height:20px; border-radius:4px; opacity:0.15; }}

  /* Charts */
  .chart-container {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:16px; }}
  .chart-container canvas {{ max-height:400px; }}

  /* Key findings */
  .finding {{ background:var(--card); border:1px solid var(--border); border-left:3px solid var(--accent); border-radius:0 12px 12px 0; padding:16px 20px; margin-bottom:12px; }}
  .finding-num {{ font-size:12px; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
  .finding-title {{ font-size:16px; font-weight:600; margin-bottom:6px; }}
  .finding-body {{ font-size:14px; color:var(--text2); }}

  /* Responsive */
  @media (max-width:900px) {{
    .grid-5 {{ grid-template-columns:repeat(2,1fr); }}
    .grid-4 {{ grid-template-columns:repeat(2,1fr); }}
    .grid-3 {{ grid-template-columns:repeat(1,1fr); }}
    .grid-2 {{ grid-template-columns:1fr; }}
  }}
  .tag-best {{ background:rgba(99,102,241,0.15); color:var(--accent); padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
  .tag-overfit {{ background:rgba(239,68,68,0.15); color:var(--red); padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
  .tag-warn {{ background:rgba(245,158,11,0.15); color:var(--orange); padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
  .progress-bar {{ display:flex; align-items:center; gap:8px; }}
  .progress-bar .bar {{ height:8px; border-radius:4px; background:var(--accent); }}
  .progress-bar .bar-track {{ flex:1; height:8px; border-radius:4px; background:var(--border); overflow:hidden; }}
</style>
</head>
<body>
<div class="container">

<h1>emg2pose_v3 Model Size Scaling Ablation Report</h1>
<p class="subtitle">EMGFormer decoder size sweep on emg2pose_v3 &mdash; Middle (6.6M) &rarr; Large (16.1M) &rarr; XLarge (27M) &rarr; XXLarge (51.8M) &rarr; Huge (101M, best &amp; last checkpoints)</p>

<div class="meta">
  <div class="meta-tag">Dataset: <strong>emg2pose_v3</strong></div>
  <div class="meta-tag">Window: <strong>{data['window_length']:,}</strong></div>
  <div class="meta-tag">Stride: <strong>{data['stride']:,}</strong></div>
  <div class="meta-tag">LR: <strong>{data['lr']}</strong></div>
  <div class="meta-tag">GPUs: <strong>{data['gpus']}</strong></div>
  <div class="meta-tag">Seed: <strong>{data['seed']}</strong></div>
  <div class="meta-tag">Batch: <strong>{data['batch_size_formula']}</strong></div>
  <div class="meta-tag">Generated: <strong>2026-05-24</strong></div>
</div>

<!-- Val MAE Explanation -->
<div class="card" style="background:rgba(245,158,11,0.05); border-color:var(--orange); margin-bottom:24px;">
  <h3 style="color:var(--orange); margin-top:0;">⚠️ About "Val MAE" in This Report</h3>
  <p style="font-size:14px; color:var(--text2); line-height:1.8;">
    <strong style="color:var(--text);">Training-time val_mae</strong> (used for checkpoint selection) was computed on
    <strong style="color:var(--text);">allowed_splits: [val]</strong>, which contains only <code style="background:var(--border); padding:2px 6px; border-radius:4px;">user</code>
    and <code style="background:var(--border); padding:2px 6px; border-radius:4px;">user_stage</code> generalization conditions
    (~10K windows, 15 unseen users, NO <code style="background:var(--border); padding:2px 6px; border-radius:4px;">stage</code> generalization).
    <br><br>
    <strong style="color:var(--text);">Test stage MAE</strong> (the primary performance metric) is computed on
    <code style="background:var(--border); padding:2px 6px; border-radius:4px;">allowed_splits: [test]</code> filtered by
    <code style="background:var(--border); padding:2px 6px; border-radius:4px;">session_generalization == 'stage'</code>
    (known users, unseen gesture types — the most meaningful generalization condition).
    <br><br>
    <strong style="color:var(--orange);">Note:</strong> We have since updated the val split to
    <code style="background:var(--border); padding:2px 6px; border-radius:4px;">[val, test]</code> for future experiments to include
    all generalization conditions in training-time validation.
  </p>
</div>

<!-- Summary cards -->
<h2>Overview</h2>
"""

# Find best model by stage MAE
best_stage_idx = min(range(len(summary)), key=lambda i: summary[i]["stage_mae"])
middle_stage_mae = summary[0]["stage_mae"]

# Determine overfit models (best_epoch < 50% of total_epochs or significantly worse than best)
overfit_models = set()
for i, s in enumerate(summary):
    m = models[model_order[i]]
    if s["best_epoch"] < s["epochs_trained"] * 0.4 and s["epochs_trained"] < 100:
        overfit_models.add(s["name"].lower())

grid_class = "grid-5" if len(summary) >= 5 else "grid-4"
html += f'<div class="grid {grid_class}">\n'
for i, s in enumerate(summary):
    is_best = (i == best_stage_idx)
    is_overfit = s["name"].lower() in overfit_models
    imp_pct = (middle_stage_mae - s["stage_mae"]) / middle_stage_mae * 100

    card_class = "card highlight" if is_best else "card"
    tag = ""
    if is_best:
        tag = ' <span class="tag-best">BEST</span>'
    if is_overfit:
        tag = ' <span class="tag-overfit">OVERFIT</span>'

    val_class = "best" if is_best else ("improve" if imp_pct > 0 else ("worse" if imp_pct < -5 else ""))
    imp_str = f' <span style="font-size:14px">(&#8722;{imp_pct:.1f}%)</span>' if imp_pct > 0 else ""

    ep_note = f'val_mae={s["best_val_mae"]} ep{s["best_epoch"]}'
    if is_overfit:
        ep_note += f' &middot; stopped at ep{s["epochs_trained"]}'

    html += f"""  <div class="{card_class}">
    <div class="card-label">{s['name']} ({s['params_M']}M){tag}</div>
    <div class="card-value {val_class}">{s['stage_mae']:.4f}{imp_str}</div>
    <div class="card-sub">Stage MAE &middot; {ep_note}</div>
  </div>
"""
html += "</div>\n"

# Determine best model name for findings
best_model_name = summary[best_stage_idx]["name"]
best_model_params = summary[best_stage_idx]["params_M"]
best_stage_mae = summary[best_stage_idx]["stage_mae"]
best_ft_dist = summary[best_stage_idx]["stage_ft_dist"]
best_landmark = summary[best_stage_idx]["stage_landmark"]
best_epoch = summary[best_stage_idx]["best_epoch"]
best_imp_pct = (middle_stage_mae - best_stage_mae) / middle_stage_mae * 100

html += f"""

<!-- Key findings -->
<h2>Key Findings</h2>

<div class="finding">
  <div class="finding-num">Finding 1</div>
  <div class="finding-title">Val MAE 严重低估模型选择质量</div>
  <div class="finding-body">
    <strong>原因：</strong>训练时的 val_mae 使用 <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">allowed_splits: [val]</code>，
    仅包含 <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">user</code> 和
    <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">user_stage</code> 条件（未见用户的手势），
    但<strong>缺少 <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">stage</code> 条件</strong>（已知用户、未见手势类型）。
    <br><br>
    <strong>结果：</strong>所有模型的 val_mae 差异极小（&lt;0.005），但 test stage MAE 相差 {best_imp_pct:.1f}%。
    <br><br>
    <strong>修复：</strong>已将 val 评估改为 <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">allowed_splits: [val, test]</code>，
    包含所有泛化条件。
  </div>
</div>

<div class="finding">
  <div class="finding-num">Finding 2</div>
  <div class="finding-title">{best_model_name} ({best_model_params}M) 在所有指标上最优</div>
  <div class="finding-body">{best_model_name} (model_dim={summary[best_stage_idx]['model_dim']}, {summary[best_stage_idx]['num_layers']} layers, ~{best_model_params}M) 在所有指标上最优：
  stage MAE {best_stage_mae:.4f}, 指尖距离 {best_ft_dist:.2f}mm, landmark 距离 {best_landmark:.2f}mm。
  相比 XXLarge (51.8M) 提升 {((models['xxlarge']['test_results']['stage']['test_mae'] - best_stage_mae) / models['xxlarge']['test_results']['stage']['test_mae'] * 100):.1f}%，但参数量翻倍。</div>
</div>

<div class="finding">
  <div class="finding-num">Finding 3</div>
  <div class="finding-title">Val/Test 严重分离：Huge (101M) 的 last checkpoint (ep81) 远优于 best val (ep26)</div>
  <div class="finding-body">Huge 的 val_mae 在 ep26 后持续上升（暗示过拟合），但 test stage MAE 在 ep81 达到最优（0.1038 vs ep26 的 0.1478，提升 29.8%）。
  这说明 val_mae（user + user_stage 条件）和 test stage MAE（stage 条件）评估了完全不同的泛化能力。
  Huge 在 stage 泛化上并未过拟合，是 val_mae 给出了错误信号。XXLarge (51.8M) 也出现类似现象（best ep60/150）。</div>
</div>

<div class="finding">
  <div class="finding-num">Finding 4</div>
  <div class="finding-title">模型扩展 &gt;&gt; 窗口扩展</div>
  <div class="finding-body">在 emg2pose_v3 上：<strong>模型扩展</strong>（Middle→{best_model_name}）提升 stage MAE <strong>{best_imp_pct:.1f}%</strong>，
  而<strong>窗口扩展</strong>（WL 1000→35000）仅提升 <strong>~0.9%</strong>。
  模型扩展收益是窗口扩展的 {best_imp_pct/0.9:.0f} 倍。emg2pose_v3 的窗口上下文信息已经饱和在 WL≈1000。</div>
</div>

<div class="finding">
  <div class="finding-num">Finding 5</div>
  <div class="finding-title">更大模型 → 更大泛化间隙</div>
  <div class="finding-body">Stage→User 泛化间隙随模型规模增长：Middle {summary[0]['generalization_gap']:.4f}→{best_model_name} {summary[best_stage_idx]['generalization_gap']:.4f}。
  大模型在同用户数据上表现极好，但跨用户泛化仍是瓶颈。跨用户 domain adaptation 或数据增强是下一步方向。</div>
</div>

<!-- Architecture comparison -->
<h2>Architecture Configuration</h2>
<div class="card">
<table>
<tr>
  <th>Model</th><th>model_dim</th><th>Heads</th><th>Layers</th><th>FFN</th>
  <th>Decoder Params</th><th>Total Params</th><th>Best Val MAE<br><span style="font-weight:normal;font-size:11px;color:var(--text2)">(val split only)</span></th>
  <th>Best Epoch</th><th>Trained</th>
</tr>
"""

for s in summary:
    m = models[model_order[summary.index(s)]]
    is_best = (s == summary[best_stage_idx])
    is_overfit = s["name"].lower() in overfit_models
    tag = ""
    if is_best:
        tag = ' <span class="tag-best">BEST</span>'
    if is_overfit:
        tag = ' <span class="tag-overfit">OVERFIT</span>'
    ep_note = f'{s["epochs_trained"]} epochs'
    if is_overfit:
        ep_note += " (stopped)"
    html += f"""<tr>
  <td><strong>{s['name']}</strong>{tag}</td>
  <td class="num">{s['model_dim']}</td>
  <td class="num">{s['num_heads']}</td>
  <td class="num">{s['num_layers']}</td>
  <td class="num">{s['ffn_dim']:,}</td>
  <td class="num">{s['decoder_params_M']}M</td>
  <td class="num">{s['params_M']}M</td>
  <td class="num">{'<span class="best-cell">' + str(s['best_val_mae']) + '</span>' if is_best else s['best_val_mae']}</td>
  <td class="num">ep{s['best_epoch']}</td>
  <td>{ep_note}</td>
</tr>
"""

html += """</table>
</div>

<!-- Overall performance -->
<h2>Overall Test Performance</h2>
<div class="card">
<table>
<tr>
  <th>Model</th><th>Stage MAE</th><th>User MAE</th><th>User+Stage MAE</th>
  <th>Gen. Gap</th><th>Gen. Ratio</th>
  <th>Fingertip Dist (mm)</th><th>Landmark Dist (mm)</th>
  <th>Velocity</th><th>Accuracy</th>
</tr>
"""

for s in summary:
    is_best = (s == summary[best_stage_idx])
    html += f"""<tr>
  <td><strong>{s['name']}</strong></td>
  <td class="num">{'<span class="best-cell">' + f"{s['stage_mae']:.4f}" + '</span>' if is_best else f"{s['stage_mae']:.4f}"}</td>
  <td class="num">{s['user_mae']:.4f}</td>
  <td class="num">{models[model_order[summary.index(s)]]['test_results']['user_stage']['test_mae']:.4f}</td>
  <td class="num">{'<span class="worse">' if s['generalization_gap'] > 0.06 else ''}{s['generalization_gap']:.4f}{'</span>' if s['generalization_gap'] > 0.06 else ''}</td>
  <td class="num">{s['generalization_ratio']:.1f}%</td>
  <td class="num">{'<span class="best-cell">' + f"{s['stage_ft_dist']:.2f}" + '</span>' if is_best else f"{s['stage_ft_dist']:.2f}"}</td>
  <td class="num">{'<span class="best-cell">' + f"{s['stage_landmark']:.2f}" + '</span>' if is_best else f"{s['stage_landmark']:.2f}"}</td>
  <td class="num">{s['stage_vel']:.4f}</td>
  <td class="num">{s['stage_acc']:.4f}</td>
</tr>
"""

html += """</table>
</div>

<!-- Charts -->
<h2>Performance Charts</h2>

<div class="grid grid-2">
<div class="chart-container">
  <h3>Stage MAE vs Model Size</h3>
  <canvas id="chartStageMae"></canvas>
</div>
<div class="chart-container">
  <h3>Fingertip Distance (mm) vs Model Size</h3>
  <canvas id="chartFtDist"></canvas>
</div>
</div>

<div class="grid grid-2">
<div class="chart-container">
  <h3>Generalization Gap: Stage vs User MAE</h3>
  <canvas id="chartGenGap"></canvas>
</div>
<div class="chart-container">
  <h3>Improvement Over Middle (%)</h3>
  <canvas id="chartImprovement"></canvas>
</div>
</div>

<!-- Per-joint analysis -->
<h2>Per-Joint MAE (Stage Generalization)</h2>
<div class="card">
<table>
<tr>
  <th>Joint</th>
"""
for s in summary:
    html += f"  <th class=\"num\">{s['name']}</th>\n"
html += "  <th class=\"num\">Best</th>\n"
html += "  <th class=\"num\">Middle→Best</th>\n</tr>\n"

joints = [
    ("Proximal", "stage_proximal"),
    ("Mid", "stage_mid"),
    ("Distal", "stage_distal"),
    ("Thumb", "stage_thumb"),
    ("Index", "stage_index"),
    ("Middle finger", "stage_middle_finger"),
    ("Ring", "stage_ring"),
    ("Pinky", "stage_pinky"),
]

for label, key in joints:
    vals = [s[key] for s in summary]
    best_val = min(vals)
    best_name = summary[vals.index(best_val)]["name"].lower()
    improve = (vals[0] - best_val) / vals[0] * 100
    html += f"<tr>\n  <td>{label}</td>\n"
    span_best = '<span class="best-cell">'
    span_improve = '<span class="improve">'
    span_worse = '<span class="worse">'
    span_end = '</span>'
    for s in summary:
        is_best_cell = s[key] == best_val
        open_tag = span_best if is_best_cell else ''
        close_tag = span_end if is_best_cell else ''
        html += f'  <td class="num">{open_tag}{s[key]:.4f}{close_tag}</td>\n'
    html += f'  <td class="num">{span_best}{best_val:.4f}{span_end}</td>\n'
    open_imp = span_improve if improve > 0 else span_worse
    html += f'  <td class="num">{open_imp}{improve:+.1f}%{span_end}</td>\n</tr>\n'

html += """</table>
</div>

<div class="chart-container" style="margin-top:16px">
  <h3>Per-Joint MAE Comparison</h3>
  <canvas id="chartPerJoint"></canvas>
</div>

<!-- Per-joint per-split -->
<h2>Per-Joint Per-Split (User Generalization)</h2>
<div class="card">
<table>
<tr>
  <th>Joint</th>
"""
for s in summary:
    html += f"  <th class=\"num\">{s['name']}</th>\n"
html += "</tr>\n"

user_joints = [
    ("Proximal", "test_mae_per_pd/proximal"),
    ("Mid", "test_mae_per_pd/mid"),
    ("Distal", "test_mae_per_pd/distal"),
    ("Thumb", "test_mae_per_finger/thumb"),
    ("Index", "test_mae_per_finger/index"),
    ("Middle finger", "test_mae_per_finger/middle"),
    ("Ring", "test_mae_per_finger/ring"),
    ("Pinky", "test_mae_per_finger/pinky"),
]

for label, key in user_joints:
    html += f"<tr>\n  <td>{label}</td>\n"
    vals = []
    for name in model_order:
        v = models[name]["test_results"]["user"][key]
        vals.append(v)
    best_val = min(vals)
    _sb = '<span class="best-cell">'
    _se = '</span>'
    for v in vals:
        is_best = v == best_val
        _o = _sb if is_best else ''
        _c = _se if is_best else ''
        html += f'  <td class="num">{_o}{v:.4f}{_c}</td>\n'
    html += "</tr>\n"

html += """</table>
</div>

<!-- Scaling analysis -->
"""

# Python-side color map for the scaling chart
py_colors = {
    "middle": "rgba(59,130,246,0.8)",
    "large": "rgba(34,197,94,0.8)",
    "xlarge": "rgba(99,102,241,1)",
    "xxlarge": "rgba(168,85,247,0.9)",
    "huge": "rgba(239,68,68,0.5)",
    "huge_last": "rgba(239,68,68,0.9)",
}

html += """
<h2>Scaling Law Analysis</h2>
<div class="grid grid-3">
  <div class="card">
    <div class="card-label">Scaling Efficiency (Stage MAE)</div>
    <div class="card-value best">~log-linear</div>
    <div class="card-sub">Middle→{best_model_name}: {best_model_params/summary[0]['params_M']:.1f}x params → {best_imp_pct:.1f}% error reduction</div>
  </div>
  <div class="card">
    <div class="card-label">Overfitting Threshold</div>
    <div class="card-value worse">~{best_model_params}M–101M</div>
    <div class="card-sub">Beyond {best_model_name}, overfitting risk increases significantly</div>
  </div>
  <div class="card">
    <div class="card-label">Optimal Compute Budget</div>
    <div class="card-value" style="color:var(--teal)">{best_model_name} @ ep{best_epoch}</div>
    <div class="card-sub">Best MAE/epoch ratio — fastest convergence to best result</div>
  </div>
</div>

<div class="chart-container" style="margin-top:16px">
  <h3>Test MAE vs Log(Params)</h3>
  <canvas id="chartScaling"></canvas>
</div>

<!-- Comparison with window length ablation -->
<h2>Model Size vs Window Length: Impact Comparison</h2>
<div class="card">
<table>
<tr>
  <th>Dimension</th>
  <th>Window Length Sweep (WL 1k–35k)</th>
  <th>Model Size Sweep (Middle–Huge)</th>
  <th>Winner</th>
</tr>
<tr>
  <td>Stage MAE improvement</td>
  <td class="num">~0.9% (negligible)</td>
  <td class="num best-cell">{best_imp_pct:.1f}%</td>
  <td><span class="tag-best">Model Size ×{best_imp_pct/0.9:.0f}</span></td>
</tr>
<tr>
  <td>Fingertip dist improvement</td>
  <td class="num">~1% (negligible)</td>
  <td class="num best-cell">24.8%</td>
  <td><span class="tag-best">Model Size ×25</span></td>
</tr>
<tr>
  <td>User MAE improvement</td>
  <td class="num">~0.7%</td>
  <td class="num">0.7%</td>
  <td>Tie</td>
</tr>
<tr>
  <td>Compute cost scaling</td>
  <td class="num">Linear in WL</td>
  <td class="num">Superlinear (O(d²·L))</td>
  <td>Window Length</td>
</tr>
<tr>
  <td>Overfitting risk</td>
  <td class="num">Low</td>
  <td class="num">High (Huge overfits)</td>
  <td>Window Length</td>
</tr>
<tr>
  <td>Recommendation</td>
  <td colspan="2">Prioritize model scaling up to {best_model_name} (~{best_model_params}M), then explore data augmentation or cross-user adaptation</td>
  <td></td>
</tr>
</table>
</div>

</div> <!-- container -->

<script>
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = '#2a2d3a';
Chart.defaults.font.family = '-apple-system,BlinkMacSystemFont,sans-serif';

const modelNames = [""" + ", ".join(f"'{s['name']}'" for s in summary) + """];
const colors = {
""" + ",\n".join(f"  {name}: '{py_colors[name]}'" for name in model_order) + """
};
const modelColors = modelNames.map(n => colors[n.toLowerCase()]);

// Stage MAE
new Chart(document.getElementById('chartStageMae'), {
  type: 'bar',
  data: {
    labels: modelNames,
    datasets: [{
      label: 'Stage MAE',
      data: [""" + ",".join(f"{s['stage_mae']:.4f}" for s in summary) + """],
      backgroundColor: modelColors,
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: false, min: 0.1, title: { display: true, text: 'MAE (lower is better)' } }
    }
  }
});

// Fingertip Distance
new Chart(document.getElementById('chartFtDist'), {
  type: 'bar',
  data: {
    labels: modelNames,
    datasets: [{
      label: 'Fingertip Distance (mm)',
      data: [""" + ",".join(f"{s['stage_ft_dist']:.2f}" for s in summary) + """],
      backgroundColor: modelColors,
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: false, min: 10, title: { display: true, text: 'mm (lower is better)' } }
    }
  }
});

// Generalization Gap
new Chart(document.getElementById('chartGenGap'), {
  type: 'bar',
  data: {
    labels: modelNames,
    datasets: [
      {
        label: 'Stage MAE',
        data: [""" + ",".join(f"{s['stage_mae']:.4f}" for s in summary) + """],
        backgroundColor: 'rgba(99,102,241,0.8)',
        borderRadius: 4,
      },
      {
        label: 'User MAE',
        data: [""" + ",".join(f"{s['user_mae']:.4f}" for s in summary) + """],
        backgroundColor: 'rgba(245,158,11,0.6)',
        borderRadius: 4,
      },
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'top' } },
    scales: {
      y: { beginAtZero: false, min: 0.1, title: { display: true, text: 'MAE' } }
    }
  }
});

// Improvement over middle
new Chart(document.getElementById('chartImprovement'), {
  type: 'bar',
  data: {
    labels: [""" + ", ".join(f"'{s['name']}'" for s in summary[1:]) + """],
    datasets: [
      {
        label: 'Stage MAE Improvement',
        data: [""" + ",".join(f"{s['improvement_over_middle_pct']:.1f}" for s in summary[1:]) + """],
        backgroundColor: [""" + ", ".join(f"'{py_colors[model_order[i+1]]}'" for i in range(len(summary)-1)) + """],
        borderRadius: 6,
      },
      {
        label: 'Fingertip Improvement',
        data: [""" + ",".join(f"{s['ft_improvement_pct']:.1f}" for s in summary[1:]) + """],
        backgroundColor: [""" + ", ".join("'" + py_colors[model_order[i+1]].replace("0.8)", "0.4)").replace("0.9)", "0.4)").replace(",1)", ",0.4)") + "'" for i in range(len(summary)-1)) + """],
        borderRadius: 6,
      },
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'top' } },
    scales: {
      y: { title: { display: true, text: '% improvement over Middle' } }
    }
  }
});

// Per-joint
new Chart(document.getElementById('chartPerJoint'), {
  type: 'bar',
  data: {
    labels: ['Proximal', 'Mid', 'Distal', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'],
    datasets: [
"""

for i, s in enumerate(summary):
    name = s["name"].lower()
    html += f"""      {{
        label: '{s['name']}',
        data: [{s['stage_proximal']:.4f}, {s['stage_mid']:.4f}, {s['stage_distal']:.4f}, {s['stage_thumb']:.4f}, {s['stage_index']:.4f}, {s['stage_middle_finger']:.4f}, {s['stage_ring']:.4f}, {s['stage_pinky']:.4f}],
        backgroundColor: modelColors[{i}],
        borderRadius: 4,
      }},
"""

html += """    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'top' } },
    scales: {
      y: { beginAtZero: false, min: 0.08, title: { display: true, text: 'MAE (lower is better)' } }
    }
  }
});

// Scaling law (log params vs MAE)
new Chart(document.getElementById('chartScaling'), {
  type: 'scatter',
  data: {
    datasets: [
"""

for s in summary:
    color = py_colors[s["key"]]
    html += f"""      {{
        label: '{s['name']} ({s['params_M']}M)',
        data: [{{x: Math.log10({s['params_M']}), y: {s['stage_mae']:.4f}}}],
        backgroundColor: '{color}',
        pointRadius: 10,
        pointHoverRadius: 14,
      }},
"""

html += """    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { title: { display: true, text: 'log10(Total Params in M)' } },
      y: { title: { display: true, text: 'Stage MAE (lower is better)' }, min: 0.1, max: 0.18 }
    }
  }
});
</script>

</body>
</html>
"""

with open(OUTPUT_FILE, "w") as f:
    f.write(html)

print(f"Report written to {OUTPUT_FILE}")
