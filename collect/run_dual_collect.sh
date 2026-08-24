#!/usr/bin/env sh

# Edit parameters here, then run:
#   sh collect/run_dual_collect.sh
#
# Runtime keys:
#   r: activate teleoperation
#   s: deactivate teleoperation
#   c: start recording one trajectory
#   v: stop/save current trajectory, then home both robots when enabled
#   q: quit

PYTHON_BIN="python3"

# Robot serial numbers
FIRST_SN="Rizon4R-062116"
SECOND_SN="Rizon4R-062115"

# Data collection
SAVE_ROOT="/home/xense/haptic_exo_teleop_ws/Data/pin_0819_10hz"
SESSION_NAME=""
FPS="10"
CAMERA_DEVICE_FPS="30"

# Camera profiles configured in dual_collect_utils.py and verified on this machine:
# D415 104122061018 and D405 260322275475, both at 640x480.

# Follower-to-leader wrench feedback, applied when the collector starts.
# 0.0 disables feedback and 1.0 enables it. Pose teleoperation and saved
# ext_wrench_in_tcp data remain active in both conditions.
WRENCH_FEEDBACK_SCALE="0.0"

# Master side uses Angler encoder, slave side uses Xense.
# Hardware IDs are machine-specific and should be checked before each setup.
USE_GRIPPER="true"
SLAVE_GRIPPER_ID="8a429d6ea337"

# Dual-fingertip Xense tactile collection.
# Both tactile sensors use the shared slave-gripper connection identifier.
USE_TACTILE="true"
TACTILE_LEFT_SENSOR_SN="OG001453"
TACTILE_RIGHT_SENSOR_SN="OG001455"
TACTILE_MAC_ADDR="$SLAVE_GRIPPER_ID"

# Master Angler encoder settings.
# OPEN/CLOSE_ANGLE are measured encoder endpoints. They map linearly to the
# corresponding slave widths in meters, with out-of-range angles clamped.
ANGLER_ID="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
ANGLER_INDEX="1"
ANGLER_BAUDRATE="1000000"
ANGLER_GAP="0.002"
ANGLER_STRICT="true"
ANGLER_OPEN_ANGLE="349.102"
ANGLER_CLOSE_ANGLE="314.561"
SLAVE_OPEN_WIDTH="0.075"
SLAVE_CLOSE_WIDTH="0.001"
# Width commanded at collector startup and exit; set it to a safe task pose.
INITIAL_GRIPPER_WIDTH="0.075"

# Optional LAN interface whitelist. Leave empty to let TDK try all interfaces.
# Multiple addresses can be separated by spaces, for example:
# Both robots on the current machine share this host-side interface.
# Verify it with `ip -4 addr` if the NIC configuration changes.
NETWORK_INTERFACES="192.168.97.10"

# Runtime tuning: EPS suppresses tiny width commands; WAIT_TIME allows motion.
GRIPPER_EPS="0.0001"
GRIPPER_WAIT_TIME="0.1"
NULL_SPACE_PERIOD="0.1"

# Return both robots to the fixed initial joint pose after each saved recording.
HOME_AFTER_RECORDING="true"
# Also return both robots when the collector exits.
HOME_ON_EXIT="true"
HOME_ROBOT_IDS="1,2"
HOME_DELAY="0.5"
HOME_RETRIES="3"
HOME_RETRY_DELAY="2.0"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -- \
  "$SCRIPT_DIR/dual_collect.py" \
  -1 "$FIRST_SN" \
  -2 "$SECOND_SN" \
  --save-root "$SAVE_ROOT" \
  --fps "$FPS" \
  --camera-device-fps "$CAMERA_DEVICE_FPS" \
  --wrench-feedback-scale "$WRENCH_FEEDBACK_SCALE" \
  --use-tactile "$USE_TACTILE" \
  --use-gripper "$USE_GRIPPER" \
  --gripper-eps "$GRIPPER_EPS" \
  --gripper-wait-time "$GRIPPER_WAIT_TIME" \
  --null-space-period "$NULL_SPACE_PERIOD" \
  --home-after-recording "$HOME_AFTER_RECORDING" \
  --home-on-exit "$HOME_ON_EXIT" \
  --home-robot-ids "$HOME_ROBOT_IDS" \
  --home-delay "$HOME_DELAY" \
  --home-retries "$HOME_RETRIES" \
  --home-retry-delay "$HOME_RETRY_DELAY"

if [ -n "$SESSION_NAME" ]; then
  set -- "$@" --session-name "$SESSION_NAME"
fi

if [ "$USE_GRIPPER" = "true" ]; then
  set -- "$@" \
    --slave-gripper-id "$SLAVE_GRIPPER_ID" \
    --angler-id "$ANGLER_ID" \
    --angler-index "$ANGLER_INDEX" \
    --angler-baudrate "$ANGLER_BAUDRATE" \
    --angler-gap="${ANGLER_GAP}" \
    --angler-strict "$ANGLER_STRICT" \
    --angler-open-angle "$ANGLER_OPEN_ANGLE" \
    --angler-close-angle "$ANGLER_CLOSE_ANGLE" \
    --slave-open-width "$SLAVE_OPEN_WIDTH" \
    --slave-close-width "$SLAVE_CLOSE_WIDTH" \
    --initial-gripper-width "$INITIAL_GRIPPER_WIDTH"
fi

if [ "$USE_TACTILE" = "true" ]; then
  set -- "$@" \
    --tactile-left-sensor-sn "$TACTILE_LEFT_SENSOR_SN" \
    --tactile-right-sensor-sn "$TACTILE_RIGHT_SENSOR_SN" \
    --tactile-mac-addr "$TACTILE_MAC_ADDR"
fi

for interface in $NETWORK_INTERFACES; do
  set -- "$@" --network-interface "$interface"
done

echo "Running:"
printf '  %s' "$PYTHON_BIN"
for arg in "$@"; do
  printf ' %s' "$arg"
done
printf '\n\n'

exec "$PYTHON_BIN" "$@"
