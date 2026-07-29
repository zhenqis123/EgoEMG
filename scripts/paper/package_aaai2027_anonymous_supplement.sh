#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAPER_DIR="$REPO_ROOT/paper/aaai2027"
OUTPUT_ZIP="$PAPER_DIR/EgoEMG_anonymous_code_data.zip"
OLD_ZIP="$OUTPUT_ZIP"
STAGE_ROOT=$(mktemp -d)
STAGE="$STAGE_ROOT/package"
NEW_ZIP="$STAGE_ROOT/EgoEMG_anonymous_code_data.zip"

cleanup() {
  rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$PAPER_DIR/code_supplement_README.md"
require_file "$OLD_ZIP"

mkdir -p "$STAGE"
install -m 0644 "$REPO_ROOT/.gitignore" "$STAGE/.gitignore"
install -m 0644 "$REPO_ROOT/environment.yml" "$STAGE/environment.yml"
install -m 0644 "$REPO_ROOT/setup.cfg" "$STAGE/setup.cfg"
install -m 0644 "$REPO_ROOT/setup.py" "$STAGE/setup.py"
install -m 0644 "$PAPER_DIR/code_supplement_README.md" "$STAGE/README.md"

# Preserve the already-anonymized numeric reviewer sample from the previous
# archive; all executable sources and paper artifacts are refreshed below.
(
  cd "$STAGE"
  unzip -q "$OLD_ZIP" 'LICENSE'
  unzip -q "$OLD_ZIP" 'data/egoemg_reviewer_sample/*'
)

# The numeric sample must exercise the complete bilateral 22-DoF path. Guard
# against silently regressing to the earlier left-wrist-only package.
for hand in left right; do
  for field in pitch yaw angles_valid; do
    require_file "$STAGE/data/egoemg_reviewer_sample/mocap_${hand}_wrist_${field}.dat"
  done
done

# Package the current Python implementation without caches or generated files.
while IFS= read -r -d '' source_file; do
  relative=${source_file#"$REPO_ROOT/"}
  install -D -m 0644 "$source_file" "$STAGE/$relative"
done < <(
  find "$REPO_ROOT/emg2pose" -type f \
    \( -name '*.py' -o -name '*.json' -o -name 'LICENSE' \) \
    -not -path '*/__pycache__/*' -print0
)

# Keep the active layered configuration tree while excluding archived sweeps.
mkdir -p "$STAGE/config"
rsync -a \
  --exclude '_archive/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$REPO_ROOT/config/" "$STAGE/config/"

mkdir -p "$STAGE/assets"
for asset in \
  emg_norm_stats.npz \
  per_dataset_norm_stats.json \
  per_dataset_norm_stats_repro_filtered_paper_alias.json; do
  require_file "$REPO_ROOT/assets/$asset"
  install -m 0644 "$REPO_ROOT/assets/$asset" "$STAGE/assets/$asset"
done

mkdir -p "$STAGE/scripts/eval" "$STAGE/scripts/data" "$STAGE/scripts/experiments"
for script in \
  unified_center_eval.py \
  analyze_per_gesture.py; do
  require_file "$REPO_ROOT/scripts/eval/$script"
  install -m 0644 "$REPO_ROOT/scripts/eval/$script" "$STAGE/scripts/eval/$script"
done
require_file "$REPO_ROOT/scripts/data/build_egoemg_vision_index.py"
install -m 0644 \
  "$REPO_ROOT/scripts/data/build_egoemg_vision_index.py" \
  "$STAGE/scripts/data/build_egoemg_vision_index.py"
for script in \
  run_5vision_s_simple_frozen_30e_20260727.sh \
  run_remaining_simple_fusion_30e_20260728.sh \
  run_vitb_vitl_wilor_tf_long_6gpu.sh \
  train_wilor_s_simple_unfrozen_lr1e-5_6gpu_20260728.sh; do
  require_file "$REPO_ROOT/scripts/run/$script"
  install -m 0755 "$REPO_ROOT/scripts/run/$script" "$STAGE/scripts/experiments/$script"
done

# Rewrite machine-specific locations in the staged copy only. These portable
# locations are documented placeholders and do not alter experiment semantics.
while IFS= read -r -d '' text_file; do
  perl -pi -e 's#data/experiment_inputs/vision_checkpoints#checkpoints/vision#g;' \
    -e 's#${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}#.#g;' \
    -e 's#${WILOR_PATH:-../WiLoR}#third_party/WiLoR#g;' \
    -e 's#${WILOR_PATH:-../WiLoR}#third_party/WiLoR#g;' \
    -e 's#data/WiLoR#third_party/WiLoR#g;' \
    -e 's#data#data#g;' \
    -e 's#/share/being-h/xizh#data#g;' \
    -e 's#python#python#g;' \
    -e 's#${HOME}/miniconda3/etc/profile\.d/conda\.sh#\$\{CONDA_EXE%/bin/conda\}/etc/profile.d/conda.sh#g;' \
    -e 's#xiziheng#anonymous#g;' \
    -e 's#being-h#anonymous-storage#g;' \
    "$text_file"
done < <(
  find "$STAGE" -type f \
    \( -name '*.py' -o -name '*.json' -o -name '*.md' -o -name '*.yaml' \
       -o -name '*.yml' -o -name '*.sh' -o -name '*.cfg' \) -print0
)

# All 2x7 fusion-ablation configurations must be explicit in the archive.
for backbone in rn18 rn50 rn152 vits vitb vitl wilor; do
  for state in frozen unfrozen; do
    require_file "$STAGE/config/experiment/fusion/fusion_${backbone}_s_simple_${state}_augbest_30e.yaml"
  done
done

# Parse every packaged Python file and reject direct author/machine identifiers.
python - "$STAGE" <<'PY'
import ast
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
for path in root.rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

patterns = [
    re.compile(value, re.IGNORECASE)
    for value in (
        r"xiziheng",
        r"being-h",
        r"/home/",
        r"/mnt/nvme/",
        r"/share/",
    )
]
hits = []
text_suffixes = {".py", ".json", ".md", ".yaml", ".yml", ".sh", ".cfg"}
for path in root.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in text_suffixes:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in patterns:
        if pattern.search(text):
            hits.append(f"{path.relative_to(root)}: {pattern.pattern}")
if hits:
    raise SystemExit("Anonymity scan failed:\n" + "\n".join(hits))
PY

(
  cd "$STAGE"
  zip -q -r -9 "$NEW_ZIP" .
)
unzip -tq "$NEW_ZIP"

# Replace only after the new archive has passed all checks.
mv -f "$NEW_ZIP" "$OUTPUT_ZIP"
echo "Created: $OUTPUT_ZIP"
