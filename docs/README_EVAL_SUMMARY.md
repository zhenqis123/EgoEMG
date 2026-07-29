# 评估用户/阶段/用户阶段分裂的 SSL 微调模型

## 完成的工作

### 1. 创建的评估脚本和工具

| 文件 | 用途 | 状态 |
|------|------|------|
| `emg2pose/test_analysis_ssl.py` | SSL 模型评估主脚本 | ✅ 完成 |
| `config/experiment/eval_ssl_ft.yaml` | Hydra 评估配置 | ✅ 完成 |
| `scripts/evaluate_ssl_ft.sh` | 单次评估 Bash 封装 | ✅ 完成 |
| `scripts/evaluate_all_ssl_checkpoints.sh` | 批量评估脚本 | ✅ 完成 |
| `scripts/compare_ssl_to_baseline.py` | 结果对比工具 | ✅ 完成 |

### 2. 创建的文档

| 文件 | 内容 | 状态 |
|------|------|------|
| `docs/EVALUATION_GUIDE.md` | 完整评估使用指南 | ✅ 完成 |
| `docs/ssl_evaluation_protocol.md` | 评估协议详细说明 | ✅ 完成 |
| `docs/ssl_eval_design.md` | 评估设计文档 | ✅ 完成 |
| `docs/ssl_eval_results.md` | 结果记录模板 | ✅ 完成 |
| `experiment_status.md` | 已更新评估计划部分 | ✅ 完成 |

### 3. 数据分割设计

评估基于 `metadata.csv` 中的 `generalization` 列，将测试集分为三个互斥的子集：

```
generalization 标注规则:
- held_out_user=True, held_out_stage=False  → user (跨用户泛化)
- held_out_user=False, held_out_stage=True  → stage (跨手势泛化)
- held_out_user=True, held_out_stage=True   → user_stage (联合泛化)
```

### 4. 使用方法

#### 快速评估最佳 checkpoint

```bash
cd . (repo root)

# 方法 1: 使用封装脚本
./scripts/evaluate_ssl_ft.sh \
    -c checkpoints/ssl_best.ckpt \
    -d $HOME/emg2pose_dataset \
    -g 0

# 方法 2: 评估所有 checkpoint
./scripts/evaluate_all_ssl_checkpoints.sh $HOME/emg2pose_dataset 0

# 方法 3: 直接 Python 调用
python -m emg2pose.test_analysis_ssl \
    checkpoint=checkpoints/ssl_best.ckpt \
    data_location=$HOME/emg2pose_dataset
```

#### 对比 baseline

```bash
python scripts/compare_ssl_to_baseline.py ssl_eval_results.csv
```

### 5. 输出格式

评估生成 CSV 文件包含三个 split 的结果：

```csv
generalization,test_mae,test_vel,test_acc,test_jerk,...
user,0.2225,0.5960,0.0085,0.0086,...
stage,0.2659,0.8499,0.0124,0.0126,...
user_stage,0.2726,0.8381,0.0123,0.0125,...
```

### 6. 待完成的任务

运行实际评估（需要 GPU 和完整数据集）：

```bash
# 评估最佳 checkpoint (约 15-20 分钟)
./scripts/evaluate_ssl_ft.sh -c checkpoints/ssl_best.ckpt

# 对比结果
python scripts/compare_ssl_to_baseline.py ssl_eval_results.csv
```

### 7. Baseline 参考值

| Split | test_mae (rad) | test_mae (deg) |
|-------|----------------|----------------|
| user | 0.2225 | 12.75° |
| stage | 0.2659 | 15.24° |
| user_stage | 0.2726 | 15.62° |

**目标**: user_stage < 0.254 rad

### 8. 关键设计决策

1. **与 baseline 相同的评估协议**: 使用相同的 `generalization` 分裂和评估指标
2. **一次评估所有三个 split**: 确保公平对比，避免 cherry-picking
3. **自动化对比工具**: `compare_ssl_to_baseline.py` 自动生成改进百分比
4. **完整的文档**: 从快速开始到详细协议，多层次文档支持

## 下一步

1. 运行评估脚本获取实际结果
2. 填充 `docs/ssl_eval_results.md` 中的结果表格
3. 更新论文的 Table 2 和 Figure 3
4. 撰写论文中的实验部分
