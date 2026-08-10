#!/usr/bin/env sh

# End-to-end Data_Collect -> ACP helper.
#
# Edit the parameters in this block, then run from Data_Collect:
#   sh postprocess/run_acp_pipeline.sh

set -e

PYTHON_BIN="python3"

# 1) Your Data_Collect session directory, or a root containing multiple sessions.
#    Multiple paths can be separated by spaces.
SESSIONS="/home/xense/haptic_exo_teleop_ws/Data/pick_0715"

# 2) Camera folder to export. Required when each session has multiple cameras.
CAMERA_NAME="cam_327322062498"

# 3) Dataset name used by ACP. The raw and zarr folders will both use this name.
DATASET_NAME="pick_0715_main_acp_v1"

# 4) Parent folder for ACP raw episodes:
#    $RAW_DATASET_PARENT/$DATASET_NAME/episode_000000/...
RAW_DATASET_PARENT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_raw"

# 5) Parent folder for ACP zarr output:
#    $PROCESSED_DATASET_PARENT/$DATASET_NAME
PROCESSED_DATASET_PARENT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_processed"

# 6) ACP repository root.
ACP_ROOT="../adaptive_compliance_policy"

# 7) Single-arm data should stay [0].
ID_LIST="[0]"

# 8) Around one second of force data. Example: force-fps 200 -> 200.
WRENCH_MOVING_AVERAGE_WINDOW_SIZE="190"

# 9) Data_Collect wrench is already recorded in the TCP frame, so keep this true.
LABEL_FLAG_REAL="true"

# 10) Label script process count. Use 1 when debugging or plotting.
LABEL_NUM_PROCESSES="1"
LABEL_FLAG_PLOT="False"

# 11) Pipeline switches.
RUN_RAW_CONVERT="true"
RUN_ZARR_CONVERT="true"
RUN_LABELS="true"

# 12) Training is long; enable only after zarr and labels are checked.
RUN_TRAIN="false"
TRAIN_CONFIG="train_conv_workspace"
CHECKPOINT_ROOT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints"

EPISODE_PREFIX="episode"
FORCE="true"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_COLLECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
ACP_ROOT_ABS=$(CDPATH= cd -- "$DATA_COLLECT_ROOT/$ACP_ROOT" && pwd)

RAW_DATASET_DIR="$RAW_DATASET_PARENT/$DATASET_NAME"
PROCESSED_DATASET_DIR="$PROCESSED_DATASET_PARENT/$DATASET_NAME"
TMP_REAL_DATA_PROCESSING="$ACP_ROOT_ABS/PyriteUtility/data_pipeline/.acp_real_data_processing_configured.py"
TMP_LABEL_SCRIPT="$ACP_ROOT_ABS/PyriteEnvSuites/scripts/.acp_postprocess_add_virtual_target_label_configured.py"

cleanup() {
  rm -f "$TMP_REAL_DATA_PROCESSING" "$TMP_LABEL_SCRIPT"
}
trap cleanup EXIT

make_tmp_real_data_processing() {
  "$PYTHON_BIN" - "$ACP_ROOT_ABS/PyriteUtility/data_pipeline/real_data_processing.py" \
    "$TMP_REAL_DATA_PROCESSING" \
    "$DATASET_NAME" \
    "$ID_LIST" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dataset_name = sys.argv[3]
id_list = sys.argv[4]

text = src.read_text(encoding="utf-8")
text = text.replace("id_list = [0]  # single robot", f"id_list = {id_list}  # configured by run_acp_pipeline.sh", 1)
text = text.replace('os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/flip_up_new_v5"', f'os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/{dataset_name}"', 1)
text = text.replace('os.environ.get("PYRITE_DATASET_FOLDERS") + "/flip_up_new_v5"', f'os.environ.get("PYRITE_DATASET_FOLDERS") + "/{dataset_name}"', 1)
dst.write_text(text, encoding="utf-8")
PY
}

make_tmp_label_script() {
  "$PYTHON_BIN" - "$ACP_ROOT_ABS/PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py" \
    "$TMP_LABEL_SCRIPT" \
    "$DATASET_NAME" \
    "$ID_LIST" \
    "$WRENCH_MOVING_AVERAGE_WINDOW_SIZE" \
    "$LABEL_NUM_PROCESSES" \
    "$LABEL_FLAG_PLOT" \
    "$LABEL_FLAG_REAL" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dataset_name = sys.argv[3]
id_list = sys.argv[4]
wrench_window = sys.argv[5]
num_processes = sys.argv[6]
flag_plot = sys.argv[7]
flag_real = sys.argv[8]

flag_real_py = "True" if flag_real.lower() in ("1", "true", "yes", "y") else "False"

text = src.read_text(encoding="utf-8")
text = text.replace('dataset_path = dataset_folder_path + "/flip_up_new_v5/"', f'dataset_path = dataset_folder_path + "/{dataset_name}/"', 1)
text = text.replace("id_list = [0]", f"id_list = {id_list}", 1)
text = text.replace("wrench_moving_average_window_size = 7000  # should be around 1s of data", f"wrench_moving_average_window_size = {wrench_window}  # configured by run_acp_pipeline.sh", 1)
text = text.replace("num_of_process = 5", f"num_of_process = {num_processes}", 1)
text = text.replace("flag_plot = False", f"flag_plot = {flag_plot}", 1)
text = text.replace(
    'flag_real = False\nif "real" in dataset_path:\n    flag_real = True',
    f"flag_real = {flag_real_py}  # configured by run_acp_pipeline.sh",
    1,
)
dst.write_text(text, encoding="utf-8")
PY
}

echo "ACP pipeline parameters:"
echo "  SESSIONS=$SESSIONS"
echo "  CAMERA_NAME=$CAMERA_NAME"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  RAW_DATASET_DIR=$RAW_DATASET_DIR"
echo "  PROCESSED_DATASET_DIR=$PROCESSED_DATASET_DIR"
echo "  ACP_ROOT=$ACP_ROOT_ABS"
echo "  WRENCH_MOVING_AVERAGE_WINDOW_SIZE=$WRENCH_MOVING_AVERAGE_WINDOW_SIZE"
echo

mkdir -p "$RAW_DATASET_PARENT" "$PROCESSED_DATASET_PARENT"

if [ "$RUN_RAW_CONVERT" = "true" ]; then
  set -- \
    "$SCRIPT_DIR/convert_acp_raw.py" \
    -o "$RAW_DATASET_DIR" \
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

  echo "Step 1/4: Data_Collect session -> ACP raw"
  "$PYTHON_BIN" "$@"
  echo
fi

export PYRITE_RAW_DATASET_FOLDERS="$RAW_DATASET_PARENT"
export PYRITE_DATASET_FOLDERS="$PROCESSED_DATASET_PARENT"
export PYRITE_CHECKPOINT_FOLDERS="$CHECKPOINT_ROOT"

if [ "$RUN_ZARR_CONVERT" = "true" ]; then
  echo "Step 2/4: ACP raw -> zarr"
  make_tmp_real_data_processing
  (
    cd "$ACP_ROOT_ABS"
    "$PYTHON_BIN" "$TMP_REAL_DATA_PROCESSING"
  )
  echo
fi

if [ "$RUN_LABELS" = "true" ]; then
  echo "Step 3/4: Add virtual target and stiffness labels"
  make_tmp_label_script
  (
    cd "$ACP_ROOT_ABS"
    "$PYTHON_BIN" "$TMP_LABEL_SCRIPT"
  )
  echo
fi

if [ "$RUN_TRAIN" = "true" ]; then
  echo "Step 4/4: Train ACP"
  mkdir -p "$CHECKPOINT_ROOT"
  (
    cd "$ACP_ROOT_ABS/PyriteML"
    HYDRA_FULL_ERROR=1 accelerate launch train.py \
      --config-name="$TRAIN_CONFIG" \
      task.dataset_path="$PROCESSED_DATASET_DIR"
  )
else
  echo "Step 4/4: training skipped. Set RUN_TRAIN=\"true\" after checking the zarr dataset."
fi

cat <<EOF

Done.

Generated or expected paths:
  ACP raw:   $RAW_DATASET_DIR
  ACP zarr:  $PROCESSED_DATASET_DIR

For training later:
  cd $ACP_ROOT_ABS/PyriteML
  export PYRITE_DATASET_FOLDERS=$PROCESSED_DATASET_PARENT
  export PYRITE_CHECKPOINT_FOLDERS=$CHECKPOINT_ROOT
  HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=$TRAIN_CONFIG task.dataset_path=$PROCESSED_DATASET_DIR
EOF
