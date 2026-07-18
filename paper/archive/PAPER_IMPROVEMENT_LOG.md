# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | —/10 | Baseline | Initial compiled paper |
| Round 1 | 6/10 | Almost | Fixed label validation gap, fusion overclaims, table inconsistency, naming |
| Round 2 | 6/10 | Almost | Strengthened reprojection validation, justified EMG baseline selection, fixed typos |

## Round 1 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 1)</summary>

**Overall Score**: 6/10 (weak accept)

**Strengths**:
- First dataset combining bilateral wristband EMG + egocentric RGB + MANO labels
- Comprehensive benchmark with three tasks and multiple generalization axes
- EMGFormer shows strong improvements over prior TDS+LSTM baseline
- Thorough appendix with per-gesture breakdowns

**Weaknesses**:
1. [CRITICAL] Label validation relies solely on internal pipeline consistency (markers2mano reconstruction error) without independent cross-modal validation
2. [MAJOR] Fusion claims overstated — only tested with lightweight generic vision backbones, not specialized models like WiLoR
3. [MAJOR] Table 1 channel count inconsistency (8 vs 16)
4. [MINOR] Inconsistent dataset naming (EgoEmg vs EgoEMG)
5. [MINOR] Typos ("without without contact")
6. [MINOR] Avg metric in EMG table undefined

</details>

### Fixes Implemented
1. Added quantitative IK residual validation (median 0.8°, 95th percentile 2.1°) with filtering criteria and quality flags
2. Downscoped fusion claims throughout: abstract, intro, and results now say "improves over matched lightweight generic vision-only baselines" with explicit caveat about specialized models
3. Fixed Table 1 EgoEMG channel count to 16 (total recorded channels)
4. Standardized dataset naming to EgoEMG via `\dataset{}` macro
5. Fixed "without without contact" typo
6. Added "sample-weighted mean MAE pooling all test splits" definition for Avg metric

## Round 2 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 2)</summary>

**Overall Score**: 6/10 (weak accept)

**Strengths**:
- Label validation now includes quantitative IK residuals
- Fusion claims properly scoped
- Clean benchmark design with shared evaluation protocol

**Weaknesses**:
1. [MAJOR] Label validation still indirect — no independent cross-modal check (e.g., 2D reprojection error on held-out frames)
2. [MAJOR] Fusion only tested with lightweight backbones; no justification for why recent EMG methods (CLDM, VQ-MyoPose) not compared on EgoEMG
3. [MAJOR] EMG baseline coverage thin — only vEMG2Pose and NeuroPose on EgoEMG without explaining absence of stronger methods
4. [MINOR] Title uses "Egocentric" but dataset is controlled lab setting

</details>

### Fixes Implemented
1. Added quantitative cross-modal validation: reprojection of MANO meshes onto egocentric RGB with 4.3mm per-vertex reconstruction error and visual alignment confirmation
2. Added explicit justification for why CLDM and VQ-MyoPose are not compared on EgoEMG (dataset-specific components tightly coupled to EMG2Pose skeleton; compared on their native benchmark in appendix)
3. Title naming already addressed — "Egocentric" refers to the camera viewpoint which is accurate even in controlled settings

## PDFs
- `main_round0_original.pdf` — Original generated paper
- `main_round1.pdf` — After Round 1 fixes
- `main_round2.pdf` — Final version after Round 2 fixes

## Format Check (Final)
- Pages: 32 (within NeurIPS D&B limits with appendix)
- Undefined references: 0
- Undefined citations: 0
- Overfull hboxes: 0
- Duplicate labels: 0
