#!/usr/bin/env python3
"""Estimate demo force/torque thresholds from Data_Collect or converted FoAR data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datacollect_io import MAIN_CAMERA_NAME, camera_arrays, discover_sessions, percentile_dict, session_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate force and torque thresholds from demos.")
    parser.add_argument("--source", required=True, type=Path, help="Data_Collect record/root or FoAR dataset root")
    parser.add_argument("--format", choices=["auto", "datacollect", "foar"], default="auto")
    parser.add_argument("--camera-name", default=MAIN_CAMERA_NAME, help="Data_Collect camera folder for raw demo windows")
    parser.add_argument("--cam-id", default=MAIN_CAMERA_NAME.removeprefix("cam_"), help="FoAR camera id for contact windows")
    parser.add_argument("--num-obs-force", type=int, default=100)
    parser.add_argument("--num-action-force", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def detect_format(source: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (source / "train").is_dir() or (source / "val").is_dir() or (source / "calib").is_dir():
        return "foar"
    return "datacollect"


def make_frame_ids_ms(timestamps_s: np.ndarray, start_s: float) -> np.ndarray:
    frame_ids = np.rint((timestamps_s.astype(np.float64) - start_s) * 1000.0).astype(np.int64)
    for i in range(1, len(frame_ids)):
        if frame_ids[i] <= frame_ids[i - 1]:
            frame_ids[i] = frame_ids[i - 1] + 1
    return frame_ids


def load_datacollect(source: Path, camera_name: str) -> tuple[list[dict], np.ndarray, list[tuple[list[int], np.ndarray]]]:
    rows = []
    windows = []
    all_wrench = []
    for session in discover_sessions(source):
        arrays = session_arrays(session)
        cam = camera_arrays(session, camera_name)
        wrench = arrays["wrench"][:, :6]
        all_wrench.append(wrench)
        start_s = float(min(cam["camera_ts"][0], arrays["wrench_ts"][0]))
        frame_ids = make_frame_ids_ms(cam["camera_ts"], start_s).tolist()
        high_ts_ms = (arrays["wrench_ts"].astype(np.float64) - start_s) * 1000.0
        high = np.concatenate([wrench, high_ts_ms[:, None]], axis=1)
        windows.append((frame_ids, high))
        rows.append({"episode": str(session), "samples": int(len(wrench)), "frames": int(len(frame_ids))})
    if not all_wrench:
        raise RuntimeError("No wrench data found")
    return rows, np.concatenate(all_wrench, axis=0), windows


def numeric_ids(path: Path, suffix: str) -> list[int]:
    if not path.is_dir():
        return []
    return sorted(int(p.stem) for p in path.glob(f"*{suffix}") if p.stem.isdigit())


def contact_ratio_for_episode(frame_ids: list[int], high_freq: np.ndarray, force_th: float, torque_th: float, num_obs_force: int, num_action_force: int) -> tuple[int, int]:
    timestamps = high_freq[:, -1]
    force_norm = np.linalg.norm(high_freq[:, :3], axis=1)
    torque_norm = np.linalg.norm(high_freq[:, 3:6], axis=1)
    positives = 0
    total = max(0, len(frame_ids) - 1)
    for cur_idx in range(total):
        cur_force_idx = int(np.argmin(np.abs(timestamps - frame_ids[cur_idx])))
        begin = max(0, cur_force_idx - num_obs_force + 1)
        end = min(len(timestamps), cur_force_idx + num_action_force + 1)
        force_window = force_norm[begin:end]
        torque_window = torque_norm[begin:end]
        positives += int(bool((len(force_window) and force_window.max() > force_th) or (len(torque_window) and torque_window.max() > torque_th)))
    return positives, total


def load_foar(source: Path, cam_id: str) -> tuple[list[dict], np.ndarray, list[tuple[list[int], np.ndarray]]]:
    rows = []
    windows = []
    all_wrench = []
    for split in ["train", "val", "all"]:
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        for ep in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            high_path = ep / "high_freq_data" / "force_torque_tcp_joint_timestamp.npy"
            if not high_path.is_file():
                continue
            high = np.load(high_path)
            wrench = high[:, :6]
            all_wrench.append(wrench)
            frame_ids = numeric_ids(ep / f"cam_{cam_id}" / "color", ".png")
            if frame_ids:
                windows.append((frame_ids, high))
            rows.append({"episode": str(ep), "split": split, "samples": int(len(wrench)), "frames": int(len(frame_ids))})
    if not all_wrench:
        raise RuntimeError(f"No FoAR high_freq_data found under {source}")
    return rows, np.concatenate(all_wrench, axis=0), windows


def window_contact_ratio(windows: list[tuple[list[int], np.ndarray]], force_th: float, torque_th: float, args: argparse.Namespace) -> dict:
    positive = 0
    total = 0
    for frame_ids, high in windows:
        pos, cnt = contact_ratio_for_episode(frame_ids, high, force_th, torque_th, args.num_obs_force, args.num_action_force)
        positive += pos
        total += cnt
    return {
        "positive_windows": int(positive),
        "total_windows": int(total),
        "ratio": float(positive / total) if total else None,
    }


def plot_distribution(force_norm: np.ndarray, torque_norm: np.ndarray, out_path: Path, force_th: float, torque_th: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(force_norm, bins=80, alpha=0.85)
    axes[0].axvline(force_th, color="red", linestyle="--", label=f"p95={force_th:.3f}")
    axes[0].set_title("Force norm distribution")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("samples")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].hist(torque_norm, bins=80, alpha=0.85)
    axes[1].axvline(torque_th, color="red", linestyle="--", label=f"p95={torque_th:.3f}")
    axes[1].set_title("Torque norm distribution")
    axes[1].set_xlabel("Nm")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    fmt = detect_format(source, args.format)
    if fmt == "datacollect":
        episodes, wrench, windows = load_datacollect(source, args.camera_name)
    else:
        episodes, wrench, windows = load_foar(source, args.cam_id)
    force_norm = np.linalg.norm(wrench[:, :3], axis=1)
    torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1)
    force_percentiles = percentile_dict(force_norm)
    torque_percentiles = percentile_dict(torque_norm)
    recommended_force = force_percentiles["95"]
    recommended_torque = torque_percentiles["95"]

    candidates = []
    for p in ["95", "97.5", "99"]:
        f = force_percentiles[p]
        t = torque_percentiles[p]
        candidates.append(
            {
                "percentile": p,
                "force_threshold": f,
                "torque_threshold": t,
                "sample_ratio_above": float(np.mean((force_norm > f) | (torque_norm > t))),
                "window_contact_ratio": window_contact_ratio(windows, f, t, args),
            }
        )

    result = {
        "source": str(source),
        "format": fmt,
        "episodes": episodes,
        "total_samples": int(len(wrench)),
        "force_norm_percentiles": force_percentiles,
        "torque_norm_percentiles": torque_percentiles,
        "recommended": {
            "strategy": "p95 candidate; inspect p97.5/p99 and window_contact_ratio before training",
            "demo_force_threshold": recommended_force,
            "demo_torque_threshold": recommended_torque,
            "window_contact_ratio": window_contact_ratio(windows, recommended_force, recommended_torque, args),
        },
        "candidate_thresholds": candidates,
    }

    out_dir = args.output_dir or (source / "threshold_report" if source.is_dir() else source.parent / "threshold_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(force_norm, torque_norm, out_dir / "force_torque_threshold_distribution.png", recommended_force, recommended_torque)
    report_path = out_dir / "threshold_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["recommended"], indent=2, ensure_ascii=False))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
