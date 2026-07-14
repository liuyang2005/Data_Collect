#!/usr/bin/env python3
"""Convert Data_Collect sessions to FoAR RealWorldDataset layout.

Default behavior intentionally uses only the main camera cam_327322062498 and ignores wrist cameras.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from datacollect_io import MAIN_CAMERA_NAME, discover_sessions, nearest_indices, select_camera_dir, sorted_numeric_files

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_COLLECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CALIB_DIR = DATA_COLLECT_ROOT / "calib" / "data"
FOAR_INHAND_CAM_ID = "043322070878"
TCP_TO_FOAR_SENSOR_ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
FOAR_GRIPPER_DECODE_WIDTH_M = 0.095


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Data_Collect record_* sessions to FoAR format.")
    parser.add_argument("--source", required=True, type=Path, help="record_* session or directory containing sessions")
    parser.add_argument("--output", required=True, type=Path, help="FoAR dataset output root")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR, help="Directory with intrinsics.txt and extrinsics.txt")
    parser.add_argument("--source-camera", default=MAIN_CAMERA_NAME, help="Source camera folder; defaults to main view")
    parser.add_argument("--foar-camera-id", default=None, help="Camera id in FoAR output; default strips cam_ prefix")
    parser.add_argument("--calib-name", default="datacollect_calib")
    parser.add_argument("--image-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--val-count", type=int, default=None, help="Default: 1 when >1 sessions else 0")
    parser.add_argument("--wrench-frame", choices=["tcp", "foar_raw"], default="tcp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def make_frame_ids(timestamps_s: np.ndarray, start_s: float) -> np.ndarray:
    frame_ids = np.rint((np.asarray(timestamps_s, dtype=np.float64) - start_s) * 1000.0).astype(np.int64)
    if len(frame_ids) == 0:
        raise ValueError("empty camera timestamps")
    for i in range(1, len(frame_ids)):
        if frame_ids[i] <= frame_ids[i - 1]:
            frame_ids[i] = frame_ids[i - 1] + 1
    return frame_ids


def tcp_xyzw_to_foar_wxyz(tcps: np.ndarray) -> np.ndarray:
    if tcps.ndim != 2 or tcps.shape[1] < 7:
        raise ValueError(f"tcps.npy must have at least 7 columns, got {tcps.shape}")
    out = np.empty((tcps.shape[0], 7), dtype=np.float64)
    out[:, :3] = tcps[:, :3]
    out[:, 3] = tcps[:, 6]
    out[:, 4:7] = tcps[:, 3:6]
    return out


def convert_wrench_for_foar_projector(wrench: np.ndarray, wrench_frame: str) -> np.ndarray:
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"wrench must have shape (T, 6), got {wrench.shape}")
    out = np.asarray(wrench, dtype=np.float64).copy()
    if wrench_frame == "tcp":
        out[:, :3] = (TCP_TO_FOAR_SENSOR_ROT @ out[:, :3].T).T
        out[:, 3:] = (TCP_TO_FOAR_SENSOR_ROT @ out[:, 3:].T).T
    elif wrench_frame != "foar_raw":
        raise ValueError(wrench_frame)
    return out


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(mode)


def intrinsics_matrix_from_datacollect(calib_dir: Path) -> np.ndarray:
    values = np.loadtxt(calib_dir / "intrinsics.txt", dtype=np.float64).reshape(-1)
    if values.shape[0] != 4:
        raise ValueError(f"Expected intrinsics.txt as [cx, cy, fx, fy], got {values}")
    cx, cy, fx, fy = values
    return np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)


def write_calibration(output_root: Path, calib_dir: Path, target_camera_id: str, calib_name: str) -> None:
    require_file(calib_dir / "intrinsics.txt")
    require_file(calib_dir / "extrinsics.txt")
    c2b = np.loadtxt(calib_dir / "extrinsics.txt", dtype=np.float64)
    if c2b.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsics.txt, got {c2b.shape}")
    target_cam_to_base_for_foar = np.linalg.inv(c2b)
    calib_path = output_root / "calib" / calib_name
    calib_path.mkdir(parents=True, exist_ok=True)
    np.save(calib_path / "extrinsics.npy", {FOAR_INHAND_CAM_ID: np.eye(4), target_camera_id: target_cam_to_base_for_foar})
    np.save(calib_path / "tcp.npy", np.array([0.0, 0.077, 0.2665, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64))
    np.save(calib_path / "intrinsics.npy", {target_camera_id: intrinsics_matrix_from_datacollect(calib_dir)})


def split_sessions(sessions: list[Path], val_count: int | None) -> list[tuple[str, Path]]:
    if val_count is None:
        val_count = 1 if len(sessions) > 1 else 0
    if val_count < 0:
        raise ValueError("--val-count must be non-negative")
    if len(sessions) > 1 and val_count >= len(sessions):
        raise ValueError(f"--val-count {val_count} leaves no training sessions")
    val_start = len(sessions) - val_count
    return [("val" if val_count and i >= val_start else "train", session) for i, session in enumerate(sessions)]


def load_json_if_present(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_session(session: Path, split: str, output_root: Path, source_camera_name: str, target_camera_id: str | None, calib_name: str, image_mode: str, wrench_frame: str) -> dict:
    source_cam_dir = select_camera_dir(session, source_camera_name)
    source_camera_id = source_cam_dir.name.removeprefix("cam_").removesuffix("_wrist")
    target_camera_id = target_camera_id or source_camera_id

    color_files = sorted_numeric_files(source_cam_dir / "color")
    depth_files = sorted_numeric_files(source_cam_dir / "depth")
    cam_ts = np.load(source_cam_dir / "timestamps_host_s.npy")
    tcps = np.load(session / "tcps.npy")
    tcp_ts = np.load(session / "tcps_timestamps_host_s.npy")
    angles = np.load(session / "angles.npy") if (session / "angles.npy").is_file() else None
    angles_ts = np.load(session / "angles_timestamps_host_s.npy") if (session / "angles_timestamps_host_s.npy").is_file() else tcp_ts
    wrench = np.load(session / "ext_wrench_in_tcp.npy")
    wrench_ts = np.load(session / "ext_wrench_in_tcp_timestamps_host_s.npy")

    if not (len(color_files) == len(depth_files) == len(cam_ts)):
        raise RuntimeError(f"Camera count mismatch in {session.name}: color={len(color_files)}, depth={len(depth_files)}, ts={len(cam_ts)}")
    if len(tcps) != len(tcp_ts):
        raise RuntimeError(f"TCP count mismatch in {session.name}: tcps={len(tcps)}, ts={len(tcp_ts)}")
    if len(wrench) != len(wrench_ts):
        raise RuntimeError(f"Wrench count mismatch in {session.name}: wrench={len(wrench)}, ts={len(wrench_ts)}")
    if angles is not None and len(angles) != len(angles_ts):
        raise RuntimeError(f"Angles count mismatch in {session.name}: angles={len(angles)}, ts={len(angles_ts)}")

    start_s = float(min(cam_ts[0], tcp_ts[0], wrench_ts[0]))
    frame_ids = make_frame_ids(cam_ts, start_s)
    cam_robot_idx = nearest_indices(tcp_ts, cam_ts)
    force_robot_idx = nearest_indices(tcp_ts, wrench_ts)
    force_angle_idx = nearest_indices(angles_ts, wrench_ts) if angles is not None else None
    tcp_wxyz = tcp_xyzw_to_foar_wxyz(tcps)

    episode_out = output_root / split / session.name
    cam_out = episode_out / f"cam_{target_camera_id}"
    color_out = cam_out / "color"
    depth_out = cam_out / "depth"
    tcp_out = cam_out / "tcp"
    gripper_out = cam_out / "gripper_command"
    for path in [color_out, depth_out, tcp_out, gripper_out, episode_out / "high_freq_data"]:
        path.mkdir(parents=True, exist_ok=True)

    for src_color, src_depth, frame_id, robot_idx in zip(color_files, depth_files, frame_ids, cam_robot_idx):
        copy_or_link(src_color, color_out / f"{int(frame_id)}.png", image_mode)
        copy_or_link(src_depth, depth_out / f"{int(frame_id)}.png", image_mode)
        np.save(tcp_out / f"{int(frame_id)}.npy", tcp_wxyz[robot_idx].astype(np.float32))
        width_m = float(tcps[robot_idx, 7]) if tcps.shape[1] >= 8 else 0.0
        gripper_ticks = np.clip(width_m / FOAR_GRIPPER_DECODE_WIDTH_M * 1000.0, 0.0, 1000.0)
        np.save(gripper_out / f"{int(frame_id)}.npy", np.array([gripper_ticks], dtype=np.float32))

    wrench_for_foar = convert_wrench_for_foar_projector(wrench, wrench_frame)
    tcp_at_force = tcp_wxyz[force_robot_idx]
    joints_at_force = angles[force_angle_idx, :7] if angles is not None else np.zeros((len(wrench_ts), 7), dtype=np.float64)
    high_freq_timestamp_ms = (wrench_ts.astype(np.float64) - start_s) * 1000.0
    high_freq = np.concatenate([wrench_for_foar, tcp_at_force, joints_at_force, high_freq_timestamp_ms[:, None]], axis=1)
    np.save(episode_out / "high_freq_data" / "force_torque_tcp_joint_timestamp.npy", high_freq.astype(np.float32))

    metadata = load_json_if_present(session / "metadata.json")
    metadata["finish_time"] = int(frame_ids[-1])
    metadata["foar_conversion"] = {
        "source_session": str(session),
        "source_camera": source_cam_dir.name,
        "foar_camera_id": target_camera_id,
        "calib_name": calib_name,
        "image_mode": image_mode,
        "ignored_wrist_camera": True,
        "frame_id_unit": "relative_episode_milliseconds_rounded_from_host_time",
        "tcp_quaternion_order_saved_for_foar": "[qw, qx, qy, qz]",
        "gripper_command_units_saved_for_foar": "ticks decoded by FoAR as ticks / 1000 * 0.095m",
        "wrench_source": "Data_Collect ext_wrench_in_tcp.npy",
        "wrench_frame_argument": wrench_frame,
        "episode_start_host_s": start_s,
    }
    with (episode_out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    (episode_out / "timestamp.txt").write_text(f"{calib_name}\n", encoding="utf-8")

    duration_s = float(cam_ts[-1] - cam_ts[0]) if len(cam_ts) > 1 else 0.0
    return {
        "split": split,
        "episode": session.name,
        "source_camera": source_cam_dir.name,
        "foar_camera_id": target_camera_id,
        "frames": int(len(frame_ids)),
        "tcp_samples": int(len(tcps)),
        "wrench_samples": int(len(wrench)),
        "duration_s": duration_s,
        "camera_hz": (len(frame_ids) - 1) / duration_s if duration_s > 0 else None,
        "finish_time": int(frame_ids[-1]),
    }


def main() -> None:
    args = parse_args()
    sessions = discover_sessions(args.source)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    first_cam = select_camera_dir(sessions[0], args.source_camera)
    target_camera_id = args.foar_camera_id or first_cam.name.removeprefix("cam_").removesuffix("_wrist")
    write_calibration(output, args.calib_dir.expanduser().resolve(), target_camera_id, args.calib_name)

    summaries = [
        convert_session(session, split, output, args.source_camera, target_camera_id, args.calib_name, args.image_mode, args.wrench_frame)
        for split, session in split_sessions(sessions, args.val_count)
    ]
    with (output / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"source": str(args.source), "output": str(output), "sessions": summaries}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote FoAR dataset: {output}")
    print(f"Camera id: {target_camera_id}")
    for item in summaries:
        hz = "n/a" if item["camera_hz"] is None else f"{item['camera_hz']:.2f} Hz"
        print(f"[{item['split']}] {item['episode']}: frames={item['frames']}, tcp={item['tcp_samples']}, wrench={item['wrench_samples']}, camera={hz}")


if __name__ == "__main__":
    main()
