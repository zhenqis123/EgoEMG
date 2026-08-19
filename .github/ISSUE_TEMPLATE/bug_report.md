---
name: Bug report
about: Installation, training/evaluation, or visualization problems
labels: bug
---

**Environment**
- egoemg version (`python -c "import egoemg; print(egoemg.__version__)"`):
- Python version and OS:
- CPU-only or CUDA (torch version, driver):

**Public entrypoint and command**

Which documented entrypoint were you running (e.g. `scripts/viz/visualize_dataset.py vision`, `python -m egoemg.train experiment=...`)? Paste the exact command.

**Observed behavior**

Error output or unexpected result:

```
<paste output>
```

**Expected behavior**

What the documentation said should happen:

**Asset availability**

For visualization/evaluation: do you already have the required memmap,
all-intra videos, calibration, MANO assets, and precomputed crop LMDB files
(see `docs/ASSET_SETUP.md` and `docs/PRERELEASE_LIMITATIONS.md`)? Note that
research configs referencing `logs/`, `test_results/`, or `../WiLoR` paths are
not part of the public support surface.
