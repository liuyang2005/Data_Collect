#!/usr/bin/env bash

# push_0728 -> causal gripper-command labels -> ACP raw/zarr/virtual-target labels.
# The v1 feedback-state dataset and original pipeline are left untouched.

set -euo pipefail

PYTHON_BIN="/home/xense/miniconda3/envs/pyrite/bin/python"
ACCELERATE_BIN="/home/xense/miniconda3/envs/pyrite/bin/accelerate"
SESSIONS="/home/xense/haptic_exo_teleop_ws/Data/push_0728"
CAMERA_NAME="cam_260322274925_wrist"
DATASET_NAME="push_0728_gripper_command_acp_v2"

RAW_DATASET_PARENT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_raw"
PROCESSED_DATASET_PARENT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_processed"
CHECKPOINT_ROOT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints"
ACP_ROOT="../adaptive_compliance_policy_extend"

GRIPPER_CLOSE_THRESHOLD="0.065"
GRIPPER_OPEN_THRESHOLD="0.075"
GRIPPER_MOTION_SPEED_THRESHOLD_M_S="0.002"
GRIPPER_MOTION_MAX_GAP_S="0.08"
GRIPPER_MOTION_SEARCH_WINDOW_S="0.75"
GRIPPER_MOTION_MIN_DELTA_M="0.003"
GRIPPER_COMMAND_LATENCY_S="0.05"
GRIPPER_SMOOTHING_WINDOW="5"

WRENCH_MOVING_AVERAGE_WINDOW_SIZE="190"
WRENCH_OFFSET_SAMPLES="200"
WRENCH_FS="190"
ID_LIST="0"

RUN_RAW_CONVERT="true"
RUN_ZARR_CONVERT="true"
RUN_LABELS="true"
RUN_TRAIN="false"
FORCE="true"
TRAIN_CONFIG="train_gripper_conv_workspace"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_COLLECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
ACP_ROOT_ABS=$(CDPATH= cd -- "$DATA_COLLECT_ROOT/$ACP_ROOT" && pwd)
RAW_DATASET_DIR="$RAW_DATASET_PARENT/$DATASET_NAME"
PROCESSED_DATASET_DIR="$PROCESSED_DATASET_PARENT/$DATASET_NAME"

mkdir -p "$RAW_DATASET_PARENT" "$PROCESSED_DATASET_PARENT" "$CHECKPOINT_ROOT"

echo "Dataset:  $DATASET_NAME"
echo "Camera:   $CAMERA_NAME"
echo "ACP copy: $ACP_ROOT_ABS"

if [[ "$RUN_RAW_CONVERT" == "true" ]]; then
  command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/convert_acp_gripper_command_raw.py"
    --output "$RAW_DATASET_DIR"
    --camera-name "$CAMERA_NAME"
    --gripper-close-threshold "$GRIPPER_CLOSE_THRESHOLD"
    --gripper-open-threshold "$GRIPPER_OPEN_THRESHOLD"
    --motion-speed-threshold-m-s "$GRIPPER_MOTION_SPEED_THRESHOLD_M_S"
    --motion-max-gap-s "$GRIPPER_MOTION_MAX_GAP_S"
    --motion-search-window-s "$GRIPPER_MOTION_SEARCH_WINDOW_S"
    --motion-min-delta-m "$GRIPPER_MOTION_MIN_DELTA_M"
    --command-latency-s "$GRIPPER_COMMAND_LATENCY_S"
    --smoothing-window "$GRIPPER_SMOOTHING_WINDOW"
    --require-tactile
  )
  if [[ "$FORCE" == "true" ]]; then
    command+=(--force)
  fi
  read -r -a session_paths <<< "$SESSIONS"
  command+=("${session_paths[@]}")
  echo "Step 1/4: Data_Collect -> ACP raw with causal gripper-command labels"
  "${command[@]}"
fi

if [[ "$RUN_ZARR_CONVERT" == "true" ]]; then
  command=(
    "$PYTHON_BIN" "$ACP_ROOT_ABS/PyriteUtility/data_pipeline/real_data_processing_gripper.py"
    --input "$RAW_DATASET_DIR"
    --output "$PROCESSED_DATASET_DIR"
    --id-list "$ID_LIST"
    --wrench-fs "$WRENCH_FS"
  )
  if [[ "$FORCE" == "true" ]]; then
    command+=(--force)
  fi
  echo "Step 2/4: ACP raw -> zarr"
  (cd "$ACP_ROOT_ABS" && "${command[@]}")
fi

if [[ "$RUN_LABELS" == "true" ]]; then
  echo "Step 3/4: zero-offset wrench -> virtual target/stiffness labels"
  (
    cd "$ACP_ROOT_ABS"
    "$PYTHON_BIN" PyriteEnvSuites/scripts/postprocess_add_virtual_target_label_gripper.py \
      --dataset "$PROCESSED_DATASET_DIR" \
      --id-list "$ID_LIST" \
      --wrench-window "$WRENCH_MOVING_AVERAGE_WINDOW_SIZE" \
      --offset-samples "$WRENCH_OFFSET_SAMPLES"
  )
fi

export PYRITE_RAW_DATASET_FOLDERS="$RAW_DATASET_PARENT"
export PYRITE_DATASET_FOLDERS="$PROCESSED_DATASET_PARENT"
export PYRITE_CHECKPOINT_FOLDERS="$CHECKPOINT_ROOT"

if [[ "$RUN_TRAIN" == "true" ]]; then
  echo "Step 4/4: train 20D ACP with inferred command labels"
  (
    cd "$ACP_ROOT_ABS/PyriteML"
    HYDRA_FULL_ERROR=1 "$ACCELERATE_BIN" launch train.py \
      --config-name="$TRAIN_CONFIG" \
      task.dataset_path="$PROCESSED_DATASET_DIR"
  )
else
  echo "Step 4/4: training skipped; inspect v2 labels before training."
fi

echo "Raw:       $RAW_DATASET_DIR"
echo "Processed: $PROCESSED_DATASET_DIR"
echo "Summary:   $RAW_DATASET_DIR/gripper_command_summary.json"
echo "Config:    $TRAIN_CONFIG"
