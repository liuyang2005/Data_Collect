#!/usr/bin/env python3
"""Create after-collection plots for one Data_Collect record_* session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datacollect_io import (
    MAIN_CAMERA_NAME,
    WRIST_CAMERA_NAME,
    alignment_stats_ms,
    camera_arrays,
    estimate_rate_hz,
    nearest_indices,
    percentile_dict,
    session_arrays,
)
from series_plotter import plot_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Data_Collect session signals and timestamp alignment.")
    parser.add_argument("--session", required=True, type=Path, help="record_* session directory")
    parser.add_argument("--camera-name", default=MAIN_CAMERA_NAME, help="Camera folder to visualize; defaults to main view")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <session>/visualization")
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows instead of only saving files")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def build_series(arrays: dict) -> list[dict]:
    tcps = arrays["tcps"]
    angles = arrays["angles"]
    wrench = arrays["wrench"]
    tcp_ts = arrays["tcp_ts"]
    angle_ts = arrays["angle_ts"]
    wrench_ts = arrays["wrench_ts"]

    series = [
        {"data": tcps[:, :3], "timestamps": tcp_ts, "labels": ["x", "y", "z"], "name": "TCP position"},
        {"data": tcps[:, 3:7], "timestamps": tcp_ts, "labels": ["qx", "qy", "qz", "qw"], "name": "TCP quaternion"},
    ]
    if tcps.shape[1] >= 8:
        series.append({"data": tcps[:, 7], "timestamps": tcp_ts, "labels": ["width_m"], "name": "Gripper width from tcps"})
    joint_cols = min(7, angles.shape[1])
    series.append({"data": angles[:, :joint_cols], "timestamps": angle_ts, "labels": [f"q{i+1}" for i in range(joint_cols)], "name": "Joint angles"})
    if angles.shape[1] >= 8:
        series.append({"data": angles[:, 7], "timestamps": angle_ts, "labels": ["width_m"], "name": "Gripper width from angles"})
    series.extend(
        [
            {"data": wrench[:, :3], "timestamps": wrench_ts, "labels": ["fx", "fy", "fz"], "name": "Force in TCP"},
            {"data": wrench[:, 3:6], "timestamps": wrench_ts, "labels": ["tx", "ty", "tz"], "name": "Torque in TCP"},
            {
                "data": np.column_stack([np.linalg.norm(wrench[:, :3], axis=1), np.linalg.norm(wrench[:, 3:6], axis=1)]),
                "timestamps": wrench_ts,
                "labels": ["force_norm", "torque_norm"],
                "name": "Force / torque norm",
            },
        ]
    )
    return series


def plot_alignment(out_path: Path, cam: dict, arrays: dict, dpi: int) -> dict:
    camera_ts = cam["camera_ts"]
    tcp_ts = arrays["tcp_ts"]
    wrench_ts = arrays["wrench_ts"]
    angle_ts = arrays["angle_ts"]

    tcp_idx = nearest_indices(tcp_ts, camera_ts)
    wrench_idx = nearest_indices(wrench_ts, camera_ts)
    angle_idx = nearest_indices(angle_ts, camera_ts)
    err_tcp_ms = (camera_ts - tcp_ts[tcp_idx]) * 1000.0
    err_wrench_ms = (camera_ts - wrench_ts[wrench_idx]) * 1000.0
    err_angle_ms = (camera_ts - angle_ts[angle_idx]) * 1000.0
    frame = np.arange(len(camera_ts))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), squeeze=False)
    ax = axes[0, 0]
    ax.plot(frame, err_tcp_ms, label="camera -> tcp", linewidth=1.0)
    ax.plot(frame, err_wrench_ms, label="camera -> wrench", linewidth=1.0)
    ax.plot(frame, err_angle_ms, label="camera -> angles", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Nearest-neighbor alignment error per camera frame")
    ax.set_xlabel("camera frame index")
    ax.set_ylabel("signed error (ms)")
    ax.grid(True, alpha=0.28)
    ax.legend()

    ax = axes[1, 0]
    bins = 40
    ax.hist(np.abs(err_tcp_ms), bins=bins, alpha=0.55, label="camera -> tcp")
    ax.hist(np.abs(err_wrench_ms), bins=bins, alpha=0.55, label="camera -> wrench")
    ax.hist(np.abs(err_angle_ms), bins=bins, alpha=0.55, label="camera -> angles")
    ax.set_title("Absolute alignment error distribution")
    ax.set_xlabel("absolute error (ms)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "camera_to_tcp": alignment_stats_ms(tcp_ts, camera_ts),
        "camera_to_wrench": alignment_stats_ms(wrench_ts, camera_ts),
        "camera_to_angles": alignment_stats_ms(angle_ts, camera_ts),
    }


def plot_sampling_intervals(out_path: Path, cam: dict, arrays: dict, dpi: int) -> None:
    streams = [
        ("camera", cam["camera_ts"]),
        ("tcp", arrays["tcp_ts"]),
        ("angles", arrays["angle_ts"]),
        ("wrench", arrays["wrench_ts"]),
    ]
    fig, axes = plt.subplots(len(streams), 1, figsize=(14, 8), squeeze=False)
    for ax, (name, ts) in zip(axes.ravel(), streams):
        if len(ts) < 2:
            ax.set_title(f"{name}: not enough samples")
            continue
        intervals_ms = np.diff(ts) * 1000.0
        ax.plot(intervals_ms, linewidth=0.9)
        ax.set_title(f"{name} sample interval, rate={estimate_rate_hz(ts) or 0:.2f} Hz")
        ax.set_ylabel("ms")
        ax.grid(True, alpha=0.28)
    axes.ravel()[-1].set_xlabel("sample index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def summary(session: Path, cam: dict, arrays: dict, alignment: dict) -> dict:
    wrench = arrays["wrench"]
    force_norm = np.linalg.norm(wrench[:, :3], axis=1)
    torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1)
    all_cams = sorted(p.name for p in session.iterdir() if p.is_dir() and p.name.startswith("cam_"))
    return {
        "session": str(session),
        "camera_used": cam["camera_name"],
        "all_camera_dirs": all_cams,
        "wrist_camera_present": WRIST_CAMERA_NAME in all_cams,
        "counts": {
            "color_files": len(cam["color_files"]),
            "depth_files": len(cam["depth_files"]),
            "camera_timestamps": len(cam["camera_ts"]),
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
        "alignment_ms": alignment,
        "force_norm_percentiles": percentile_dict(force_norm),
        "torque_norm_percentiles": percentile_dict(torque_norm),
        "foar_required_data_present": {
            "main_rgb": bool(cam["color_files"]),
            "main_depth": bool(cam["depth_files"]),
            "tcp": arrays["tcps"].shape[1] >= 7,
            "gripper_width": arrays["tcps"].shape[1] >= 8 or arrays["angles"].shape[1] >= 8,
            "joint_angles": arrays["angles"].shape[1] >= 7,
            "force_torque": arrays["wrench"].shape[1] >= 6,
        },
    }


def main() -> None:
    args = parse_args()
    session = args.session.expanduser().resolve()
    output_dir = args.output_dir or (session / "visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = session_arrays(session)
    cam = camera_arrays(session, args.camera_name)

    plot_timeseries(
        build_series(arrays),
        title=f"Data_Collect session: {session.name}",
        time_unit="s",
        save_path=output_dir / "timeseries_overview.png",
        show=args.show,
        dpi=args.dpi,
    )
    alignment = plot_alignment(output_dir / "timestamp_alignment.png", cam, arrays, args.dpi)
    plot_sampling_intervals(output_dir / "sampling_intervals.png", cam, arrays, args.dpi)

    report = summary(session, cam, arrays, alignment)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote visualization report: {output_dir}")
    print(f"Main plots: {output_dir / 'timeseries_overview.png'}, {output_dir / 'timestamp_alignment.png'}")


if __name__ == "__main__":
    main()
