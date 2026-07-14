#!/usr/bin/env python3
"""Report timestamp alignment quality for Data_Collect sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datacollect_io import (
    MAIN_CAMERA_NAME,
    alignment_stats_ms,
    camera_arrays,
    discover_sessions,
    estimate_rate_hz,
    nearest_indices,
    session_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Data_Collect timestamp alignment.")
    parser.add_argument("--source", required=True, type=Path, help="record_* session or directory containing sessions")
    parser.add_argument("--camera-name", default=MAIN_CAMERA_NAME)
    parser.add_argument("--output", type=Path, default=None, help="JSON output path. Default writes beside the session/root.")
    parser.add_argument("--no-write", action="store_true", help="Only print report, do not write JSON")
    return parser.parse_args()


def session_report(session: Path, camera_name: str) -> dict:
    arrays = session_arrays(session)
    cam = camera_arrays(session, camera_name)
    camera_ts = cam["camera_ts"]
    tcp_ts = arrays["tcp_ts"]
    angle_ts = arrays["angle_ts"]
    wrench_ts = arrays["wrench_ts"]

    cam_to_tcp_idx = nearest_indices(tcp_ts, camera_ts)
    cam_to_force_idx = nearest_indices(wrench_ts, camera_ts)
    force_to_tcp_idx = nearest_indices(tcp_ts, wrench_ts)

    return {
        "session": str(session),
        "camera_name": cam["camera_name"],
        "counts": {
            "camera_frames": int(len(camera_ts)),
            "color_files": int(len(cam["color_files"])),
            "depth_files": int(len(cam["depth_files"])),
            "tcp_samples": int(len(tcp_ts)),
            "angle_samples": int(len(angle_ts)),
            "wrench_samples": int(len(wrench_ts)),
        },
        "rates_hz": {
            "camera": estimate_rate_hz(camera_ts),
            "tcp": estimate_rate_hz(tcp_ts),
            "angles": estimate_rate_hz(angle_ts),
            "wrench": estimate_rate_hz(wrench_ts),
        },
        "alignment_ms": {
            "camera_to_tcp": alignment_stats_ms(tcp_ts, camera_ts),
            "camera_to_wrench": alignment_stats_ms(wrench_ts, camera_ts),
            "wrench_to_tcp": alignment_stats_ms(tcp_ts, wrench_ts),
            "angles_to_tcp": alignment_stats_ms(tcp_ts, angle_ts),
        },
        "nearest_index_ranges": {
            "camera_to_tcp": [int(cam_to_tcp_idx.min()), int(cam_to_tcp_idx.max())] if len(cam_to_tcp_idx) else [],
            "camera_to_wrench": [int(cam_to_force_idx.min()), int(cam_to_force_idx.max())] if len(cam_to_force_idx) else [],
            "wrench_to_tcp": [int(force_to_tcp_idx.min()), int(force_to_tcp_idx.max())] if len(force_to_tcp_idx) else [],
        },
        "monotonic": {
            "camera_ts": bool(np.all(np.diff(camera_ts) > 0)) if len(camera_ts) > 1 else True,
            "tcp_ts": bool(np.all(np.diff(tcp_ts) > 0)) if len(tcp_ts) > 1 else True,
            "angle_ts": bool(np.all(np.diff(angle_ts) > 0)) if len(angle_ts) > 1 else True,
            "wrench_ts": bool(np.all(np.diff(wrench_ts) > 0)) if len(wrench_ts) > 1 else True,
        },
    }


def default_output(source: Path, sessions: list[Path]) -> Path:
    if len(sessions) == 1:
        out_dir = sessions[0] / "visualization"
        return out_dir / "alignment_report.json"
    return source.expanduser().resolve() / "alignment_summary.json"


def main() -> None:
    args = parse_args()
    sessions = discover_sessions(args.source)
    reports = [session_report(session, args.camera_name) for session in sessions]
    result = {"source": str(args.source), "sessions": reports}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if not args.no_write:
        output = args.output or default_output(args.source, sessions)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
