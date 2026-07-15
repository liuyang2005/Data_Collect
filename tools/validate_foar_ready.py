#!/usr/bin/env python3
"""Check whether Data_Collect sessions have the data needed for FoAR conversion/training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datacollect_io import MAIN_CAMERA_NAME, camera_arrays, discover_sessions, estimate_rate_hz, session_arrays

DEFAULT_CALIB_DIR = Path(__file__).resolve().parents[1] / "calib" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Data_Collect sessions for FoAR conversion readiness.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--camera-name", default=MAIN_CAMERA_NAME)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def check_session(session: Path, camera_name: str) -> dict:
    issues = []
    arrays = session_arrays(session)
    cam = camera_arrays(session, camera_name)
    if len(cam["color_files"]) == 0:
        issues.append("missing main RGB images")
    if len(cam["depth_files"]) == 0:
        issues.append("missing main depth images")
    if len(cam["color_files"]) != len(cam["depth_files"]):
        issues.append("RGB/depth file counts differ")
    if len(cam["color_files"]) != len(cam["camera_ts"]):
        issues.append("camera image count differs from timestamps_host_s.npy")
    if arrays["tcps"].ndim != 2 or arrays["tcps"].shape[1] < 7:
        issues.append("tcps.npy must contain at least [x,y,z,qx,qy,qz,qw]")
    if arrays["tcps"].shape[1] < 8 and arrays["angles"].shape[1] < 8:
        issues.append("missing gripper width column")
    if arrays["angles"].ndim != 2 or arrays["angles"].shape[1] < 7:
        issues.append("angles.npy must contain 7 joint angles")
    if arrays["wrench"].ndim != 2 or arrays["wrench"].shape[1] < 6:
        issues.append("ext_wrench_in_tcp.npy must contain 6D force/torque")
    for name, data_key, ts_key in [
        ("tcp", "tcps", "tcp_ts"),
        ("angles", "angles", "angle_ts"),
        ("wrench", "wrench", "wrench_ts"),
    ]:
        if len(arrays[data_key]) != len(arrays[ts_key]):
            issues.append(f"{name} data length differs from timestamp length")
    for name, ts_key in [("camera", None), ("tcp", "tcp_ts"), ("angles", "angle_ts"), ("wrench", "wrench_ts")]:
        ts = cam["camera_ts"] if ts_key is None else arrays[ts_key]
        if len(ts) > 1 and not np.all(np.diff(ts) > 0):
            issues.append(f"{name} timestamps are not strictly increasing")
    return {
        "session": str(session),
        "ready": not issues,
        "issues": issues,
        "camera_used": cam["camera_name"],
        "counts": {
            "color": len(cam["color_files"]),
            "depth": len(cam["depth_files"]),
            "camera_ts": len(cam["camera_ts"]),
            "tcps": len(arrays["tcps"]),
            "angles": len(arrays["angles"]),
            "wrench": len(arrays["wrench"]),
        },
        "rates_hz": {
            "camera": estimate_rate_hz(cam["camera_ts"]),
            "tcp": estimate_rate_hz(arrays["tcp_ts"]),
            "angles": estimate_rate_hz(arrays["angle_ts"]),
            "wrench": estimate_rate_hz(arrays["wrench_ts"]),
        },
    }


def main() -> None:
    args = parse_args()
    sessions = discover_sessions(args.source)
    calib_dir = args.calib_dir.expanduser().resolve()
    calib_issues = []
    if not (calib_dir / "intrinsics.txt").is_file():
        calib_issues.append(f"missing {calib_dir / 'intrinsics.txt'}")
    if not (calib_dir / "extrinsics.txt").is_file():
        calib_issues.append(f"missing {calib_dir / 'extrinsics.txt'}")
    session_reports = [check_session(session, args.camera_name) for session in sessions]
    result = {
        "source": str(args.source),
        "camera_name": args.camera_name,
        "calib_dir": str(calib_dir),
        "calibration_ready": not calib_issues,
        "calibration_issues": calib_issues,
        "sessions": session_reports,
        "foar_training_required_data": [
            "main-view RGB images",
            "main-view depth images",
            "camera intrinsics/extrinsics calibration",
            "TCP pose [x,y,z,qx,qy,qz,qw]",
            "gripper width to generate FoAR gripper_command",
            "7 robot joint angles",
            "6D force/torque wrench and timestamps",
        ],
        "not_required_by_current_foar_dataset": ["TCP velocity", "wrist camera"],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
