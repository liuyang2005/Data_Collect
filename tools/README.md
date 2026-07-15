# Data_Collect Tools

These tools adapt the useful Visualize/SeriesPlot idea from Forcemimic2Pipeline to this repository's Data_Collect `record_*` layout.

Default camera is the main view:

```text
cam_327322062498
```

The wrist camera `cam_260322274925_wrist` is ignored by FoAR conversion unless explicitly requested.

## After-collection visualization

```bash
python tools/visualize_episode.py \
  --session /home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713/record_xxx \
  --camera-name cam_327322062498
```

Outputs under `record_xxx/visualization/`:

- `timeseries_overview.png`
- `timestamp_alignment.png`
- `sampling_intervals.png`
- `summary.json`

## Timestamp alignment report

```bash
python tools/align_datacollect_timestamps.py \
  --source /home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713/record_xxx
```

For a single session, the default output is `record_xxx/visualization/alignment_report.json`.

## Convert to FoAR format

```bash
python tools/convert_datacollect_to_foar.py \
  --source /home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713 \
  --output /home/xense/haptic_exo_teleop_ws/jiaqingke/foar_data/pick_0713 \
  --source-camera cam_327322062498 \
  --calib-dir /home/xense/haptic_exo_teleop_ws/jiaqingke/Data_Collect/calib/data \
  --overwrite
```

FoAR required data produced by the converter:

- `train|val/<episode>/cam_<id>/color/*.png`
- `train|val/<episode>/cam_<id>/depth/*.png`
- `train|val/<episode>/cam_<id>/tcp/*.npy`
- `train|val/<episode>/cam_<id>/gripper_command/*.npy`
- `train|val/<episode>/high_freq_data/force_torque_tcp_joint_timestamp.npy`
- `calib/<calib_name>/intrinsics.npy`, `extrinsics.npy`, `tcp.npy`

## Check FoAR readiness

```bash
python tools/validate_foar_ready.py \
  --source /home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713
```

Current FoAR training needs RGB, depth, calibration, TCP pose, gripper width, 7 joint angles, and 6D force/torque. It does not use TCP velocity or the wrist camera.

## Estimate force/torque thresholds

```bash
python tools/estimate_force_torque_thresholds.py \
  --source /home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713 \
  --format datacollect
```

The script reports force/torque norm percentiles and recommends p95 as a candidate threshold. Inspect p97.5/p99 and the distribution plot before using the values in FoAR training.
