## MAE 对比（rad / deg）

> 角度换算：  
> `deg = rad × 180 / π ≈ rad × 57.2958`

### Overall MAE

| Generalization | Reproduced MAE (rad) | Reproduced MAE (deg) | Reproduced MAE – Seq. Window, no `ik_failure` (rad) | Reproduced MAE – Seq. Window, no `ik_failure` (deg) | Official MAE (rad) | Official MAE (deg) |
|---------------|----------------------|----------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------|--------------------|
| user | 0.22221 | 12.73° | 0.21922 | 12.56° | 0.22246 | 12.75° |
| stage | 0.26605 | 15.24° | 0.25626 | 14.68° | 0.26593 | 15.24° |
| user_stage | 0.27135 | 15.55° | 0.26845 | 15.38° | 0.27257 | 15.62° |

| Method / Setting | MAE (deg, user_stage) |
|------------------|-----------|
| Official | 15.62° |
| Reproduced (original setting) | 15.55° |
| Reproduced (sequential sliding window, no skip `ik_failure`) | 15.38° |

---