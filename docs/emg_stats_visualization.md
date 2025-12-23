# EMG 统计与可视化说明（中文）

本文说明 `scripts/` 下 EMG 统计与可视化脚本的输出含义，默认针对完整数据集（`data/emg2pose_data`），在无头环境仅保存图像，可加 `--show` 交互显示。

## 流程
1) `emg_stats_per_file.py`：全量遍历，逐文件逐通道计算统计，输出 `logs/emg_stats_per_file.csv`。
2) `emg_stats_aggregate.py`：将 per-file 统计聚合到 user/stage/side/channel 级别，输出 `logs/emg_stats_aggregate.csv`。
3) `emg_stats_visualize.py`：从聚合 CSV 生成图表，默认保存到 `logs/emg_stats_plots/`。

## 每通道统计指标
- `mean`/`std`/`min`/`max`/`p05`/`p95`：基础分布统计。
- `rms`：均方根幅值。
- `energy`：平方和（信号能量）。
- `sum_abs`：绝对值和。
- `finite_frac`：有限值占比（过滤 NaN/Inf）。

## 生成的图
- **热图**  
  - 用户 × 通道 RMS：行=user，列=channel，值为均值 RMS，观察用户-通道能量分布。
  - 阶段 × 通道 RMS：行=stage，列=channel，值为均值 RMS，观察不同 stage 影响。
- **箱线图 / 分布图**  
  - 按通道的箱线图：`rms_mean`/`energy_mean`/`std_mean`，并有 side 分色版本。  
  - 直方图/KDE：上述指标按 side 或 stage 分色，查看分布形状差异。
- **相关性热图**  
  - 按 user/stage 聚合后的通道相关性（基于 RMS/energy 均值），展示跨通道相关结构。
- **PCA 散点**  
  - 基于多项均值特征（rms/energy/std）做 PCA，按 user 或 stage 着色，观察聚类/分离情况。

## 运行示例
- 全量统计（默认多进程加速）：  
  `python scripts/emg_stats_per_file.py --data-root data/emg2pose_data --output logs/emg_stats_per_file.csv`
- 聚合：  
  `python scripts/emg_stats_aggregate.py --input logs/emg_stats_per_file.csv --output logs/emg_stats_aggregate.csv`
- 可视化（仅保存；加 `--show` 交互显示）：  
  `python scripts/emg_stats_visualize.py --input logs/emg_stats_aggregate.csv --output-dir logs/emg_stats_plots`

## 解读提示
- 某些通道 RMS/energy 持续偏高：可能与电极位置/运动伪迹相关，可对比不同 stage/user 判断规律性。
- 通道高相关性：可能指示共模噪声或耦合肌肉激活。
- PCA 聚类：user/stage 间若分离明显，说明存在主体/任务特异的通道能量模式；重叠则更同质。
- 关注 `finite_frac`：偏低可能存在传感器掉点或预处理缺口。
