<style>
  /* 容器美化 */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 25px 0;
    font-family: "Segoe UI", Segoe, Tahoma, Arial, sans-serif;
    font-size: 14px;
    /* 采用学术论文常用的三线表风格 */
    border-top: 2px solid #2c3e50; 
    border-bottom: 2px solid #2c3e50;
  }

  /* 表头样式 */
  th {
    background-color: #f8f9fa;
    color: #34495e;
    font-weight: 700;
    padding: 12px 15px;
    border-bottom: 1px solid #bdc3c7; /* 表头下方的横线 */
    text-align: center !important;
  }

  /* 单元格样式 */
  td {
    padding: 10px 15px;
    text-align: center;
    border-bottom: 1px solid #ecf0f1; /* 极淡的行分割线 */
    color: #2c3e50;
  }

  /* 隔行变色 */
  tr:nth-child(even) {
    background-color: #fcfcfc;
  }

  /* 悬停效果：方便对比实验数据 */
  tr:hover {
    background-color: #f1faff;
  }

  /* 标题美化 */
  h3 {
    border-left: 5px solid #007acc;
    padding-left: 10px;
    color: #2c3e50;
  }
</style>

# Experiment Results

## MAE 对比（rad / deg）

> 角度换算：  
> `deg = rad × 180 / π ≈ rad × 57.2958`

### Overall MAE

| Generalization | Reproduced MAE (rad) | Reproduced MAE (deg) | Reproduced MAE – Seq. Window, no `ik_failure` (rad) | Reproduced MAE – Seq. Window, no `ik_failure` (deg) | Official MAE (rad) | Official MAE (deg) |
| -------------- | -------------------- | -------------------- | --------------------------------------------------- | --------------------------------------------------- | ------------------ | ------------------ |
| user           | 0.22221              | 12.73°               | 0.21922                                             | 12.56°                                              | 0.22246            | 12.75°             |
| stage          | 0.26605              | 15.24°               | 0.25626                                             | 14.68°                                              | 0.26593            | 15.24°             |
| user_stage     | 0.27135              | 15.55°               | 0.26845                                             | 15.38°                                              | 0.27257            | 15.62°             |

### Comparison between Different Settings

| Setting | Architecture | Parameters | MAE (deg, user_stage) | MAE (deg, user) | MAE (deg, stage) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **emg2pose** | TDS + LSTM | 6.4M | 15.62 | 12.75 | 15.24 |
| emg2tendon | VAE + Diffusion | - | 14.70 | 11.30 | 14.3 |
| VQ-MyoPose (subset) | VQ-VAE + BiGRU | - | 14.60 | **10.20** | 21.9 |
| emg2pose (w aug) | TDS + LSTM | 6.4M | 14.99 | 11.48 | 14.23 |
| emgformer_nano (w aug)| TDS + Transformer | **3.1M** | 14.82 | **11.52** | 15.22 |
| emgformer_small (w aug)| TDS + Transformer | **3.5M** | **14.53** | **11.28** | 13.92 |
| emgformer_small_middle (w aug)| TDS + Transformer | **3.5M** | 14.80 | 11.34 | 14.09 |
| emgformer_middle (w aug)| TDS + Transformer | **6.4M** | 14.80 | 11.44 | **13.79** |
| emgformer_large (w aug)| TDS + Transformer | **14.1M** | 14.96 | 11.71 | 13.77 |
| emgformer_xxlarge (w aug)| TDS + Transformer | **53.2M** | 15.31 | 11.86 | 14.00 |
| emgconformer (w/o specaug) | Conformer | 1M | 16.33 | 12.75 | 15.16 |
| emgconformer (w specaug) | Conformer | 1M | 17.51 | 13.17 | 17.00 |

### Comparison between Different Settings in small emgformer
| Setting | Architecture | Parameters | MAE (deg, user_stage) | MAE (deg, user) | MAE (deg, stage) | train_mae |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |

| emgformer_small (w small aug, 0.5 channel dropout, 0.2 decoder dropout, zero masking, 7790 window length)| TDS + Transformer | 3.5M | 14.69 | 11.42 | 14.08 |
| emgformer_small (w small aug, 0.5 channel dropout, 0.2 decoder dropout, only time masking, 7790 window length)| TDS + Transformer | 3.5M | 14.68 | 11.34 | 14.07 |
| emgformer_small (w big aug, 0.6 channel dropout, 0.2 decoder dropout, middle masking, 7790 window length)| TDS + Transformer | 3.5M | 15.00 | 11.37 | 14.57 |