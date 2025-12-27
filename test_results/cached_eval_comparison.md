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

| Setting | MAE (deg, user_stage) | MAE (deg, user) | MAE (deg, stage) |
| :--- | :---: | :---: | :---: |
| **Official (tds + LSTM) (6.4M)** | 15.62 | 12.75 | 15.24 |
| emg2tendon | 14.7 | 11.30 | 14.3 |
| Official + mask ik failure | 15.38 | - | - |
| Official + casual se block | 15.60 | 12.18 | 15.16 |
| Official + se block + mask ik failure + batch norm + *aggressive augmentation*| 14.99| 11.48| 14.23 |
| tds + transformer (3.5 M) | **14.61** | **11.28** | **14.16** |
| tds + transformer (6.4 M) | **14.61** | **11.28** | **14.16** |