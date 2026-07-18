# Citation Audit Report

**Date**: 2026-05-06
**Bib file**: references.bib
**Total entries**: 38 (27 KEEP / 9 FIX / 0 REPLACE / 1 REMOVE / 1 UNCERTAIN)

## Overall Verdict: FAIL

One hallucinated entry (`kurbis2025emg`) and critical metadata errors in `hamer` must be
resolved before submission.

---

## Priority Fixes (CRITICAL — apply before submission)

### REMOVE: `kurbis2025emg` — HALLUCINATED REFERENCE

- **Bib entry**: `@article{kurbis2025emg, title={An EMG Foundation Model for Neural Decoding}, author={Kurbis, A. G. and others}, journal={arXiv preprint}, year={2025}}`
- **Problem**: No arxiv ID provided. Arxiv search for "EMG foundation model neural decoding"
  returns zero matches. All 23 "Kurbis" authors on arxiv are Nils Kürbis (philosophical
  logic, not EMG). Semantic Scholar search also returns no matching paper.
- **Cited at**: main.tex:71 — "EMG foundation models explore large-scale pretraining, but
  primarily target gesture classification rather than continuous pose"
- **ACTION**: Remove this citation. If you need 3 EMG foundation model references,
  `fasulo2025tinymyo` and `mehlman2025scaling` suffice, or replace with a verifiable
  alternative.

### FIX: `hamer` — WRONG ARXIV ID AND WRONG AUTHORS

- **Bib entry problem**: arxiv ID `2404.04330` points to an **astrophysics paper**
  ("Hydrodynamical simulations favor a pure deflagration origin of the near-Chandrasekhar
  mass supernova remnant 3C 397"). Authors in bib (Potamias, Nguyen, Kaski, Ghorbani)
  are actually the **WiLoR authors**, not HAMER authors.
- **Correct arxiv ID**: `2312.05251`
- **Correct authors**: Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo
  Kanazawa, David Fouhey, Jitendra Malik
- **Cited at**: main.tex:52, main.tex:74
- **ACTION**: Fix arxiv URL to `https://arxiv.org/abs/2312.05251`, replace author list,
  verify title.

### FIX: `umetrack` — WRONG AUTHORS

- **Bib entry**: Han, Shangchen and Liu, Po-han and Wang, Yuzhe and Ma, Rui and Liu, Ce
- **Correct authors** (arxiv 2211.00099): Shangchen Han, Po-chen Wu, Yubo Zhang, Beibei
  Liu, Linguang Zhang, Zheng Wang
- **ACTION**: Fix author list, add arxiv URL `https://arxiv.org/abs/2211.00099`.

### FIX: `emg2tendon` — WRONG TITLE

- **Bib title**: "EMG2Tendon: Generative Modeling for EMG-Based Hand Motion Reconstruction"
- **Correct title** (arxiv 2508.08269): "emg2tendon: From sEMG Signals to Tendon Control
  in Musculoskeletal Hands"
- **Also**: Paper was accepted at RSS 2025, not just an arXiv preprint.
- **ACTION**: Fix title, update venue to RSS 2025.

### FIX: `rope` — WRONG DOI YEAR

- **Bib DOI**: `10.1016/j.neucom.2024.127063`
- **Correct DOI**: `10.1016/j.neucom.2023.127063` (year in DOI is 2023; published year
  2024 is correct)
- **ACTION**: Fix DOI.

### FIX: Incomplete Author Lists (5 entries)

These entries use "and others" instead of full author lists:

| Key | Current | Should be |
|-----|---------|------------|
| `fasulo2025tinymyo` | Fasulo, M. and others | Matteo Fasulo, Giusy Spacone, Thorir Mar Ingolfsson, Yawei Li, Luca Benini, Andrea Cossettini |
| `mehlman2025scaling` | Mehlman, N. and others | Nicholas Mehlman, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Kelvin Niu, Alexander H. Miller, Shagun Sodhani |
| `zhao2026dexemg` | Zhao, Q. and others | Qianyou Zhao, Wenqiao Li, Chiyu Wang, Kaifeng Zhang |
| `gowda2025database` | Gowda, H. T. and others | Harshavardhana T. Gowda, Neha Kaul, Carlos Carrasco, Marcus A. Battraw, Safa Amer, Saniya Kotwal, Selena Lam, Zachary McNaughton, Ferdous Rahimi, Sana Shehabi, Jonathon S. Schofield, Lee M. Miller |

Also: `mehlman2025scaling` was accepted at TMLR 2025 (not just arXiv).
`gowda2025database` is missing DOI `10.1038/s41597-025-04825-z`.

---

## Uncertain Entry

### `vqmyopose` — Cannot Verify Existence

- **Problem**: Cannot find through arxiv, Semantic Scholar, or web search. HANDS workshop
  URL (hands-workshop.org) only shows a redirect page. Authors include Dario Farina (known
  EMG researcher), suggesting the paper likely exists but is not publicly indexed.
- **Cited at**: main.tex:52, 71, 326, 999
- **ACTION**: Verify directly with the authors, or replace with a verifiable reference
  before submission.

---

## All-Clean Entries (27 entries, no action needed)

| Key | Verifications |
|-----|--------------|
| `emg2pose` | arXiv 2412.02725 confirmed. NeurIPS 2024 D&B track. |
| `ninapro` | Scientific Data 2014. DOI 10.1038/sdata.2014.53. |
| `myoki` | Scientific Data 2025. DOI verified (200 OK). |
| `jarque2019` | Scientific Data 2019. DOI 10.1038/s41597-019-0285-1. |
| `grabmyo` | Scientific Data 2022. DOI 10.1038/s41597-022-01836-y verified (200). |
| `putemg` | Sensors 2019. DOI 10.3390/S19163548. |
| `du2017surface` | Sensors 2017. DOI 10.3390/s17030458. |
| `amma2015advancing` | ACM CHI 2015. DOI 10.1145/2702123.2702501. |
| `farina2014` | J Appl Physiol 2014. Well-known review. |
| `freihand` | ICCV 2019. |
| `dexycb` | ICRA 2022. DOI 10.1109/ICRA46639.2022.9812138. |
| `interhand` | TPAMI 2023. |
| `gigahand` | arXiv 2412.04244, 2024. |
| `hot3d` | ECCV 2024. arXiv 2406.09598. |
| `h2o` | ICCV 2021. DOI 10.1109/ICCV48922.2021.00998. |
| `wilor` | ECCV 2024. arXiv 2409.12259 verified. |
| `mano` | SIGGRAPH Asia 2017. DOI 10.1145/3130800.3130883. |
| `tds` | Interspeech 2019. arXiv 1904.02619. |
| `hao2024multimodal` | M2VIP 2024. DOI → IEEE Xplore verified. |
| `zandigohar2024multimodal` | Frontiers in Robotics and AI 2024. DOI verified. |
| `xi2026wristpp` | ACM CHI 2026. Author's own paper. |
| `jiang2021open` | PhysioNet 2021. |
| `liu2021neuropose` | The Web Conference 2021. |
| `zhou2019continuity` | CVPR 2019. arXiv 1812.07035. |
| `arctic` | CVPR 2023. arXiv 2204.13662. |
| `hoi4d` | CVPR 2022. arXiv 2203.01577. |
| `hanco` | DAGM GCPR 2021. |
| `ho3d` | CVPR 2020. |

---

## Context Audit Summary

All context uses for the 27 KEEP entries were verified as appropriate:
- EMG datasets cited for gesture/pose benchmarks → correct
- Vision datasets cited for hand-pose annotations → correct
- Method papers cited for their technical contributions → correct
- No wrong-context citations detected among verified entries.

## Audit Methodology

- **Existence**: Verified via arxiv API, DOI resolution (HTTP status codes), Crossref API
- **Metadata**: Cross-referenced bib entries against arxiv metadata and Crossref records
- **Context**: Manually reviewed each citation's surrounding sentence against the cited
  paper's known contributions
- **Limitations**: Some papers behind paywalls (ACM, IEEE, MDPI) could not be
  full-text-verified; well-known papers accepted on reputation.

---

Generated by `/citation-audit` skill, 2026-05-06.
