#!/usr/bin/env sh

# Edit parameters here, then run:
#   sh postprocess/run_convert_acp_raw.sh
#
# This script creates the raw input layout expected by:
#   adaptive_compliance_policy/PyriteUtility/data_pipeline/real_data_processing.py
#
# It does not create zarr directly. After this finishes, point ACP's
# real_data_processing.py at OUTPUT_RAW_DATASET and run ACP's own zarr conversion.

PYTHON_BIN="python3"

set -e

# Input can be one session directory, or a root containing multiple sessions.
# Multiple paths can be separated by spaces.
SESSIONS="/home/xense/haptic_exo_teleop_ws/liuyang/Data/pick_0701"

# Output raw dataset root for ACP. The script creates episode_000000, episode_000001, ...
OUTPUT_RAW_DATASET="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_raw_dataset"

# Camera folder to export. Required when each session has multiple cameras.
CAMERA_NAME="cam_327322062498"

EPISODE_PREFIX="episode"
FORCE="true"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -- \
  "$SCRIPT_DIR/convert_acp_raw.py" \
  -o "$OUTPUT_RAW_DATASET" \
  --episode-prefix "$EPISODE_PREFIX"

if [ -n "$CAMERA_NAME" ]; then
  set -- "$@" --camera-name "$CAMERA_NAME"
fi

if [ "$FORCE" = "true" ]; then
  set -- "$@" --force
fi

for session in $SESSIONS; do
  set -- "$@" "$session"
done

echo "Running:"
printf '  %s' "$PYTHON_BIN"
for arg in "$@"; do
  printf ' %s' "$arg"
done
printf '\n\n'

"$PYTHON_BIN" "$@"

cat <<EOF

ACP raw dataset is ready:
  $OUTPUT_RAW_DATASET

Next step in adaptive_compliance_policy:
  1. Set PYRITE_RAW_DATASET_FOLDERS to the parent of this raw dataset.
  2. Set PYRITE_DATASET_FOLDERS to the zarr output parent.
  3. In PyriteUtility/data_pipeline/real_data_processing.py, set input_dir/output_dir
     to this dataset name and your target zarr name.
  4. Run real_data_processing.py in the ACP environment.
  5. Run PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py on the zarr.
EOF
