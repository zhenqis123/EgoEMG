# EgoEMG 发布执行计划

> 状态：**代码预发布 P1 进行中**  
> 适用分支：`main`  
> 最后审查：2026-08-20  
> 目标：将仓库整理为可明确承诺、可安装、可验证、可追溯的公开 release。

本文件是执行清单，而不是对外文档。每一项完成后勾选并在 PR/commit 中
链接证据；除非对应任务明确授权，**不得删除本地数据、视频、检查点或实验结果**。
当前工作区中的 `viz_check/` 与 `zed_sample_30s.mp4` 是本地文件，不能因仓库
清理而删除。

## 0. 发布决策与冻结

**完成条件：选择唯一的发布定位，并把决定记录在本节。后续所有文案、版本和资产
均以该决定为准。**

- [x] 选择发布类型（只能选一项）。
  - [x] **A. 代码预发布**：仅发布可检查/可开发的代码和 legacy 资产，不承诺完整
    EgoEMG 数据、论文结果或完整复现。
  - [ ] **B. 正式数据发布**：发布完整数据、权重、复现说明与不可变 manifest。
- [ ] 记录目标 tag、预计日期、发布负责人和审批人。
- [x] 记录目标版本：`0.1.0rc1`；完整数据和可复现资产发布前不得使用 `1.0.0`。
- [ ] 创建 release 分支或冻结窗口；冻结后只接受 release blocker 修复。
- [ ] 为本计划建立 issue/项目看板，并为每项 P0 建立可分配任务。

验收：README、包版本、Git tag、CITATION 的发布状态不存在互相矛盾的说法。

---

## 1. 对外声明、版本与引用

### 1.1 README 发布状态

- [x] 将 README 顶部的发布状态改成与第 0 节决定完全一致的文本。
- [x] 检查并修改首页 badge：数据可用性、checkpoint 数量、Python 版本、版本号。
- [x] 检查并修改目录、Dataset、Checkpoints、Training、Evaluation、Results 各节。
- [x] 对预发布：删除或显式标为“legacy / 不保证复现”的下载、指标和命令；不得让
  读者误解为当前完整数据集。
- [ ] 对正式发布：给每个下载链接标明数据/权重版本、发布日期、许可证和适用任务。
- [x] 使 README 的视觉化示例与实际 `vision` CLI 参数、输出文件和依赖一致。
  （2026-08-20 复核：示例参数与 `visualize_dataset.py` argparse 一致；输出契约见
  `docs/PRERELEASE_LIMITATIONS.md` "Visualization contract"。）
- [x] 复查所有文档中的“released”、“reproducible”、“full release”、“latest”等
  时效性措辞。（2026-08-20 复查：仅存一处 "reproducible"，已限定为 legacy 资产
  + 对应命令；无未限定的 "full release"/"latest" 宣称。）

验收：由未参与修改的读者只看 README，能正确回答“什么已发布、什么未发布、能否复现
哪一项结果、如何获得资产”。

### 1.2 版本与 package metadata

- [x] 确定单一版本来源：`egoemg.__version__`；README badge 与 package metadata 已从
  `0.1.0rc1` 交叉校验。Git tag/release title 留待正式打包时确认。
- [x] 设置 `requires-python` 为 `>=3.10`。
- [x] 补充项目 URL、license、描述和 Python 分类器。
- [x] 用非 editable 安装验证生成的 wheel 可导入（2026-08-08）。
- [x] 确认 `find_packages()`/package data 不会遗漏 UmeTrack hand model JSON；wheel 安装后
  已验证资源存在，并添加显式 package-data/MANIFEST 规则。

验收：在干净环境运行 `pip install dist/*.whl` 后，`python -c 'import egoemg'` 成功，
且所声明的本地 package data 可被读取。

### 1.3 Citation 与论文状态

- [x] 预发布时移除 `date-released`，避免把未发布数据标成已发布；正式发布时填写真实日期。
- [x] 明确 `CITATION.cff` 当前对象是 software；正式数据发布时再增加数据引用条目。
- [ ] 论文公开后补 DOI/URL、作者列表、venue、年份、版本和 preferred citation。
- [x] 检查 CFF 语法并在 CI 中验证。（CI 步骤 "Validate citation metadata"：
  YAML 解析 + type/version/无 date-released 断言。）

验收：`CITATION.cff` 与 README 的论文/数据状态一致，且通过 CFF validator。

---

## 2. 数据、权重、许可与合规

### 2.1 资产发布清单

- [ ] 建立版本化 `release_manifest.json`（或同等格式），列出每个数据包、checkpoint、
  preview、配置和示例输出。
- [ ] 为每个文件/分卷记录：相对路径、字节数、SHA-256、数据版本、许可证、来源、用途。
- [ ] 提供独立的校验脚本，支持下载后验证完整性和缺失文件报告。
- [ ] 将 Google Drive/Baidu 的目录映射、分卷拼接、可选下载工具和失败恢复写入文档。
- [ ] 明确 legacy 包与正式包的关系；不得共用含糊的目录名或下载按钮。
- [ ] 对 checkpoint 记录：架构、输入通道、窗口长度、归一化统计、训练数据版本、配置、
  git commit、指标和许可证。
- [ ] 验证 manifest 与实际远端资产逐项一致，再发布下载链接。

验收：在一台新机器只使用公开文档和 manifest 即可下载、校验并定位每个官方资产。

### 2.2 数据伦理与访问边界

- [x] 增加数据卡占位文件，明确正式数据发布前不得对未验证事项作出声明；完整数据卡
  仍待正式数据发布时补齐。
- [ ] 写明受试者授权、隐私处理、egocentric 视频风险、允许/禁止用途、访问限制和联系渠道。
- [ ] 对 preview 和公开样例单独确认没有可识别个人信息或未授权素材。
- [ ] 明确数据许可证文本、适用范围和商业/再分发限制。

验收：数据页面和仓库中可找到完整、可执行的访问和使用条件。

### 2.3 第三方资产与许可证

- [x] 新增 `THIRD_PARTY_NOTICES.md`，逐项列出 UmeTrack、MANO、WiLoR、emg2pose、
  pretrained backbone 及其他依赖的来源、版本、许可证和再分发条件。
- [ ] 确认仓库中每个第三方源文件是否允许随源码再分发。
- [ ] 将根 MIT、数据 CC-BY-NC、衍生代码 CC-BY-NC-SA 的文件边界写清楚。
- [ ] 对 MANO/WiLoR 等不能随仓库分发的资产，提供用户自行获取步骤和运行前检查。
- [ ] 执行一次 license scan，并人工处理不兼容或未知项。

验收：用户可明确判断任一代码、数据、模型资产的许可与获取方式。

---

## 3. 安装、依赖与外部组件

### 3.1 依赖定义

- [ ] 迁移或完善为标准化的 `pyproject.toml`；保留 Conda 文件仅作为 CUDA/系统依赖方案。
- [x] 定义最小基础依赖，以及 `train`、`vision`、`viz`、`realtime`、`dev` extras。
- [x] 显式声明公开安装路径需要的第三方库，包括 `smplx`、`pyrender`、`trimesh`、
  `lmdb`、`pyarrow`、`pyzmq`、`pandas`（以最终 import audit 为准）。
- [ ] 固定或给出经过验证的版本范围；记录 CUDA/PyTorch/driver 兼容矩阵。
- [ ] 为 Linux headless/EGL、CPU-only、开发环境分别提供安装路径。
- [ ] 创建 lock file 或可导出的已验证 lock，避免 `latest` 依赖漂移。

验收：在干净 CPU 环境、干净 CUDA 环境各完成一次安装和最小 smoke test。

### 3.2 外部目录和本机路径

- [x] 扫描 `config/` 与公开 `scripts/` 中的 `../WiLoR`、`../manotorch`、`logs/`、
  `test_results/`、个人绝对路径及本机 checkpoint 路径。
- [x] 对每个引用决定：作为官方安装依赖、改为可配置环境变量、改为下载资产，或移入 archive。
  （2026-08-20 决定：audit 的 37 处引用全部位于 fusion 研究配置；一律定性为
  research record（`config/experiment/README.md` + `docs/PRERELEASE_LIMITATIONS.md`），
  路径均可用 `${oc.env:...}` 覆盖；公开 recipe 只用 legacy 资产。）
- [x] 在程序启动时做明确的路径/资产校验，报出可执行的修复建议，不能晚期以 ImportError
  或 FileNotFoundError 失败。（公开入口 `vision`：输出创建前校验 LMDB/crop key，
  缺失即报错并给出修复方式；训练入口数据依赖在 dataset 初始化时显式断言。）
- [x] 确保 README 不再将 active experiment 作为公开可运行 recipe 推荐；改为明确的
  maintainer workflow sketch，并链接 portability audit。
- [x] 将研究性脚本的默认 checkpoint 改为 `None`，或移出公开支持范围。
  （以支持面声明代替逐文件改动：`scripts/README.md` 列出公开入口，其余
  （含 `scripts/paper/` 等带本地默认路径者）明确为 research record。）

验收：新 clone 中不存在必须靠开发者同级目录或个人训练日志才能运行的“官方命令”。

### 3.3 EGL 与可视化运行环境

- [ ] 记录已支持的平台和 GPU 驱动范围，以及 EGL 为可选加速而非必要条件。
- [ ] 文档化 headless 验证命令、`PYOPENGL_PLATFORM`、GLVND 约束与 fallback 行为。
- [ ] 为 GPU EGL、OSMesa/软件渲染、无渲染依赖三种环境分别添加 smoke tests。
- [ ] 将渲染后端、GPU/CPU fallback 和实际输出编码器写入运行日志。

验收：目标服务器上能一条命令确认 GPU EGL 是否可用，并得到明确 fallback/修复信息。

---

## 4. 配置、脚本与可视化收敛

### 4.1 公开支持面的定义

- [x] 列出 release 保证支持的入口：训练、评估、下载、可视化、realtime（若公开）。
  （`scripts/README.md` "Support surface" 表：visualizer、vision index 构建、
  all-intra 重编码、portability audit、imu 修复/验证脚本。）
- [x] 将不保证维护的 notebook、分析脚本、迁移脚本和实验脚本移至明确的 `research_archive/`
  或在文件头标注 unsupported。（集中声明于 `scripts/README.md` 与
  `config/experiment/README.md`；`_archive` 配置目录已隔离。）
- [x] 保证 README 只链接公开支持的入口。（2026-08-20 复核：README 仅引用
  `scripts/viz/visualize_dataset.py` 与 `scripts/data/build_egoemg_vision_index.py`。）
- [ ] 对每个公开入口写出输入、输出、依赖、设备要求和最小示例。

验收：仓库根目录和 README 不把研究残留误呈现为正式 API。

### 4.2 可视化入口

- [ ] 确定 `vision`、`timeline`、`mesh`、`fk_vs_mano` 是否都属于 release 支持面。
- [x] 删除不可达的 `run_crops`；公开 CLI 只保留 `vision`、`timeline`、`mesh`、
  `fk_vs_mano`。
- [ ] 为 `vision` 明确输出契约：overlay MP4、左/右预裁剪 crop MP4、命名、分辨率、fps、
  stride/max-frames 语义。
- [x] 处理 crop LMDB/manifest/单手缺失/视频帧缺失：默认应失败并给出修复方式，不能静默
  输出黑色视频后返回成功。
- [x] 验证并记录 crop 视频只读取预计算 crop，而非运行时 bbox crop。
- [ ] 用小型 fixture 测试 mesh 投影、bbox、左右 crop 同步、视频时长、编码和资源释放。
- [ ] 清理 `scripts/viz/` 中已不属于公开流程的脚本，删除前确认其不承载本地工作流。

验收：`python scripts/viz/visualize_dataset.py --help`、README 示例和集成测试描述完全一致。

### 4.3 Hydra 配置

- [x] 对所有 active config 做 compose/load 测试（排除 `_archive`）。
- [ ] 保留一套每种任务的 canonical recipe，并在 README 中只引用这些 recipe。
- [x] 将 `_archive` 下的配置从 active-config composition 与 portability audit 路径隔离，
  并在配置目录中标记其历史状态。
- [ ] 校验 canonical recipe 的数据路径、checkpoint、归一化统计、设备数和结果表格相匹配。

验收：每个 README 命令可在公开依赖和对应资产齐全时启动，并使用预期配置。

---

## 5. 测试、CI 与安全

### 5.1 测试层级

- [x] 保留当前单测，并记录基线：2026-08-08 为 `42 passed`、一个 PyTorch warning。
- [x] 添加 package wheel 安装、import 和 UmeTrack package-data test。
- [x] 添加配置 compose test、README 命令 smoke test、下载 manifest validator。
  （compose：`egoemg/tests/test_active_config_composition.py`；命令 smoke：
  `test_cli_help_smoke.py` 覆盖公开入口 `--help`；manifest validator 待资产
  发布时随 §2.1 一并交付。）
- [ ] 添加 checkpoint load compatibility tests（每个官方 checkpoint 至少一个）。
- [ ] 添加 data-independent tiny fixture，覆盖训练、评估和 `vision` 视频输出。
- [ ] 建立 coverage 报告及合理的关键模块阈值。
- [x] 对已知 PyTorch warning 评估是否可消除；若保留，在 CI 中显式允许并说明原因。
  （2026-08-20 评估：warning 来自 torch TransformerEncoder 快速路径提示
  （`norm_first=True` 禁用 nested tensor），为模型结构所需，不可消除且无害；
  基线 47→49 tests 仅此一条 warning，维持显式记录，不做过滤。）

验收：测试不依赖私有数据、开发者目录或 GPU；关键公开路径都有自动化证据。

### 5.2 CI 改造

- [x] 将公开 CI 改为 CPU unit-test/packaging 环境，避免无 GPU runner 创建完整 CUDA Conda
  环境；CUDA/visualization CI 留待有可用 GPU runner 时单独添加。
- [x] 固定 GitHub Actions、Miniconda、Python 和环境依赖版本，移除无约束 `latest`。
  （actions 固定主版本 `@v4`/`@v5`；CI 已不使用 Miniconda；matrix 固定
  3.10/3.11；依赖范围见 `setup.py` extras。lock file 单列于 §3.1。）
- [x] 为 pip 设置缓存。
- [x] 让 lint/format/type checks 对真正违规失败，而非只报告 `--exit-zero`。
  （CI flake8 以 fatal 类别 E9/F63/F7/F82 失败退出；全量风格 lint 有意不在
  遗留代码上强制。）
- [x] 增加 Python 支持版本矩阵。（3.10 / 3.11。）
- [x] 增加 wheel build、安装、资源读取和测试步骤；sdist 验证留待 release CI。
- [ ] 增加 README link check、CFF validation、secret scan、dependency/license audit。
  （部分完成 2026-08-20：CFF validation 与 secret scan 已入 CI；README link
  check 与 dependency/license audit 待正式发布轮。）
- [ ] 为 tag/release 建立独立 workflow，产出可附加的 wheel、source archive、checksums 和
  测试报告。

验收：PR CI、main CI 和 release CI 各有明确职责；release workflow 可从干净 checkout 完成。

### 5.3 安全与社区文件

- [x] 新增 `SECURITY.md`，定义漏洞报告渠道与响应范围。
- [x] 新增 `CODE_OF_CONDUCT.md`、issue templates、PR template。
  （`.github/ISSUE_TEMPLATE/`：bug report + data/asset access；PR template
  含 release-plan 关联与检查项。）
- [x] 对 Git 历史和当前树运行 secret scan；发现项先轮换凭据，再移除并视需要重写历史。
  （2026-08-20 当前树扫描干净：无 token/key/密码模式命中；CI 增加常驻
  tracked-files secret scan 步骤。Git 历史深扫留待正式发布轮。）
- [x] 检查 checkpoint/data 下载链接是否暴露私人 token、共享口令或可撤销访问凭据。
  （唯一凭据为 legacy 包的百度网盘公开分享提取码——该资产的既定分发方式，
  非泄漏；无 access_token 类参数。）

验收：公开仓库具备安全联系入口，且 secret scan 结果留有记录。

---

## 6. 仓库卫生与交付物

- [x] 审核追踪的大文件（HTML report、论文 PDF、网页视频、ablation artifact 等），按“源码必需 / 
  GitHub Release 附件 / 网站部署仓库 / LFS / 删除”分类。
  （2026-08-20 审计：tracked 最大文件为 `assets/*.png` 电极/手势布局图
  （≤2.9 MB，文档引用，源码必需）；无 HTML report/PDF/视频被追踪；
  无需 LFS。本地大文件（视频/检查点/memmap）均已被 ignore 策略排除。）
- [ ] 删除或迁移前，逐个确认不属于用户本地工作流；不要使用宽泛删除命令。
- [ ] 如采用 Git LFS，安装并配置后确认远端 LFS 配额、clone 行为和 CI 拉取策略。
- [ ] 对历史中不应存在的大对象评估是否需要 history rewrite；此操作必须单独审批并提前备份。
- [x] 为 `viz_check/` 和根目录本地 MP4 制定精确 ignore 策略；本地文件仍保留在工作区。
- [ ] 清理本地 Git temporary pack garbage 前确认无进行中的 Git 操作；该步骤只影响本地 `.git`，
  不应与源码整改混在同一提交中。
- [x] 添加 `CHANGELOG.md`，记录预发布版本、已知限制和后续 release 要求。

验收：干净 clone 没有无关实验产物；仓库体积和每个大文件的存在理由可解释。

---

## 7. 正式发布演练

- [ ] 从全新目录 clone 目标 commit，不使用开发者现有 Conda 环境、数据目录或同级仓库。
- [ ] 按公开文档完成 CPU 安装；执行基础测试。
- [ ] 在独立 CUDA 机器完成 GPU 安装；执行 EGL/visualization smoke test。
- [ ] 下载 preview/正式资产（按第 0 节的发布类型），运行 manifest 校验。
- [ ] 执行一条 EMG 评估、一条 vision/fusion 评估、一条 vision 视频生成命令。
- [ ] 验证输出指标/文件与文档预期相符，记录硬件、driver、CUDA、Python、commit 和数据版本。
- [ ] 让第二位执行者独立复现一次；所有偏差必须记录为 issue 或修复。
- [ ] 运行 release CI，保存日志与生成物。

验收：独立机器上的完整演练成功，且所有命令均只依赖公开声明的输入。

---

## 8. Tag、GitHub Release 与发布后

- [ ] 确认 `git status` 干净；明确排除本地视频、数据和检查点。
- [ ] 确认所有 P0、所选支持面的 P1、发布演练均已勾选并有证据链接。
- [ ] 创建签名 tag（若团队流程要求），版本号与 package metadata 一致。
- [ ] 创建 GitHub Release：摘要、兼容性、安装方式、资产 manifest、checksums、已知限制、
  citation、license 和数据状态。
- [ ] 将 release commit、数据版本、checkpoint manifest 永久关联。
- [ ] 发布后在干净环境再次验证 GitHub 的源码包、release 附件和外部下载链接。
- [ ] 建立 issue 模板用于安装、数据访问、复现、模型兼容性和安全问题。
- [ ] 记录发布日期、最终 tag、CI run、验收负责人，并将本计划状态更新为完成。

## 完成记录

| 日期 | 项目 | 证据（PR / commit / CI / 链接） | 执行者 |
| --- | --- | --- | --- |
| 2026-08-08 | 初始 release 审查：42 tests passed；发现发布阻塞项 | 本文件创建前的审查记录 | Codex |
| 2026-08-20 | 数据修复：unified memmap EgoEMG `imu` 通道重排（41 ep / 66,161,725 行），四层验证（原始 parquet 逐位、/data 副本置换等价、数据集类冒烟、49 tests passed） | `scripts/prepare/fix_egoemg_imu_channel_order.py`、`scripts/release/imu_verify_report_windows_original.json`、`docs/data_known_issues.md` #1、CHANGELOG | ZCode |
| 2026-08-20 | release-prep 批次：社区文件、CI secret-scan、CLI smoke tests、scripts 支持面声明、措辞/大文件/链接复核、计划勾选更新 | `SECURITY.md`、`CODE_OF_CONDUCT.md`、`.github/ISSUE_TEMPLATE/`、`.github/PULL_REQUEST_TEMPLATE.md`、`.github/workflows/main.yml`、`egoemg/tests/test_cli_help_smoke.py`、`scripts/README.md` | ZCode |
| 2026-08-20 | 静态分析清理轮：包内死导入/死变量清零（42→0），CI lint 扩展至 F401/F811/F841（包内），52 处文档交叉引用全部有效，wheel 重建验证 | commit `c65029f`；flake8 双范围 0 违规；49 tests passed | ZCode |
| 2026-08-20 | 依赖声明审计：joblib/omegaconf 补入 install_requires（前者为核心 utils 模块级依赖，干净安装即炸）；其余未声明导入（av/timm/zarr/open3d/unidecode）逐一核实为惰性/受保护/惰性工厂路径 | `setup.py`；AST 导入 vs 声明审计脚本输出；`egoemg/datasets/__init__.py` 惰性工厂核实 | ZCode |
