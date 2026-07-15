#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SessionData:
    session_dir: Path
    camera_dir: Path
    color_files: list[Path]
    camera_timestamps_s: np.ndarray
    tcp_pose7_wxyz: np.ndarray
    robot_timestamps_s: np.ndarray
    wrench: np.ndarray
    wrench_timestamps_s: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Data_Collect sessions to the raw dataset layout expected by "
            "adaptive_compliance_policy/PyriteUtility/data_pipeline/real_data_processing.py."
        )
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="Session directories, or roots containing session directories.",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--camera-name",
        default="",
        help="Camera folder name, for example cam_327322062498. Required for multi-camera sessions.",
    )
    parser.add_argument(
        "--episode-prefix",
        default="episode",
        help="Output episode directory prefix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def discover_sessions(paths: Sequence[Path]) -> list[Path]:
    sessions: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if is_session_dir(path):
            sessions.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        sessions.extend(sorted(child for child in path.iterdir() if is_session_dir(child)))

    if not sessions:
        raise ValueError("no Data_Collect sessions found")
    return sessions


def is_session_dir(path: Path) -> bool:
    has_new_robot_stream = all(
        (path / "robot" / name).is_file()
        for name in ("tcp_pose.npy", "tcp_vel.npy", "q.npy", "timestamps_host_s.npy")
    )
    has_legacy_robot_stream = (
        (path / "tcps.npy").is_file()
        and (path / "tcps_timestamps_host_s.npy").is_file()
    )
    return (
        path.is_dir()
        and (has_new_robot_stream or has_legacy_robot_stream)
        and (path / "ext_wrench_in_tcp.npy").is_file()
        and (path / "ext_wrench_in_tcp_timestamps_host_s.npy").is_file()
    )


def select_camera_dir(session_dir: Path, camera_name: str = "") -> Path:
    if camera_name:
        camera_dir = session_dir / camera_name
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"camera directory not found: {camera_dir}")
        return camera_dir

    candidates = sorted(
        path
        for path in session_dir.iterdir()
        if path.is_dir() and (path / "color").is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no camera folders with color/ found in {session_dir}")
    raise ValueError(
        f"multiple camera folders found in {session_dir}: "
        f"{', '.join(path.name for path in candidates)}. Pass --camera-name."
    )


def sorted_color_files(camera_dir: Path) -> list[Path]:
    color_dir = camera_dir / "color"
    if not color_dir.is_dir():
        raise FileNotFoundError(f"color directory not found: {color_dir}")
    files = sorted(
        path for path in color_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"no color images found in {color_dir}")
    return files


def tcp_xyzw_width_to_pose7_wxyz(tcps: np.ndarray) -> np.ndarray:
    tcps = np.asarray(tcps, dtype=np.float64)
    if tcps.ndim != 2 or tcps.shape[1] != 8:
        raise ValueError(
            "tcps.npy must have shape (T, 8): [x,y,z,qx,qy,qz,qw,width], "
            f"got {tcps.shape}"
        )
    return np.column_stack([tcps[:, :3], tcps[:, 6], tcps[:, 3], tcps[:, 4], tcps[:, 5]])


def load_timestamps(path: Path, expected_len: int, name: str) -> np.ndarray:
    timestamps = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if timestamps.shape[0] != expected_len:
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, got {timestamps.shape[0]}"
        )
    if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{name} must be monotonically nondecreasing: {path}")
    return timestamps


def load_camera_timestamps_s(
    camera_dir: Path,
    color_count: int,
    robot_timestamps_s: np.ndarray,
) -> np.ndarray:
    timestamp_path = camera_dir / "timestamps_host_s.npy"
    if timestamp_path.is_file():
        return load_timestamps(timestamp_path, color_count, "camera timestamps")

    if color_count == robot_timestamps_s.shape[0]:
        return robot_timestamps_s.copy()

    raise ValueError(
        f"{camera_dir} has no timestamps_host_s.npy and image count ({color_count}) "
        f"does not match robot timestamp count ({robot_timestamps_s.shape[0]})."
    )


def load_session(session_dir: Path, camera_name: str = "") -> SessionData:
    camera_dir = select_camera_dir(session_dir, camera_name)
    color_files = sorted_color_files(camera_dir)

    robot_dir = session_dir / "robot"
    if (robot_dir / "tcp_pose.npy").is_file():
        tcp_pose_path = robot_dir / "tcp_pose.npy"
        robot_timestamps_path = robot_dir / "timestamps_host_s.npy"
    else:
        tcp_pose_path = session_dir / "tcps.npy"
        robot_timestamps_path = session_dir / "tcps_timestamps_host_s.npy"

    tcps = np.load(tcp_pose_path)
    tcp_pose7_wxyz = tcp_xyzw_width_to_pose7_wxyz(tcps)
    robot_timestamps_s = load_timestamps(
        robot_timestamps_path,
        tcp_pose7_wxyz.shape[0],
        "robot timestamps",
    )

    wrench = np.asarray(np.load(session_dir / "ext_wrench_in_tcp.npy"), dtype=np.float64)
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"ext_wrench_in_tcp.npy must have shape (T, 6), got {wrench.shape}")
    wrench_timestamps_s = load_timestamps(
        session_dir / "ext_wrench_in_tcp_timestamps_host_s.npy",
        wrench.shape[0],
        "wrench timestamps",
    )

    camera_timestamps_s = load_camera_timestamps_s(
        camera_dir,
        len(color_files),
        robot_timestamps_s,
    )

    return SessionData(
        session_dir=session_dir,
        camera_dir=camera_dir,
        color_files=color_files,
        camera_timestamps_s=camera_timestamps_s,
        tcp_pose7_wxyz=tcp_pose7_wxyz,
        robot_timestamps_s=robot_timestamps_s,
        wrench=wrench,
        wrench_timestamps_s=wrench_timestamps_s,
    )


def common_start_time_s(data: SessionData) -> float:
    first_times = [
        data.camera_timestamps_s[0],
        data.robot_timestamps_s[0],
        data.wrench_timestamps_s[0],
    ]
    return float(np.min(first_times))


def to_relative_ms(timestamps_s: np.ndarray, t0_s: float) -> np.ndarray:
    return np.round((np.asarray(timestamps_s, dtype=np.float64) - t0_s) * 1000.0, 5)


def image_name(index: int, timestamp_ms: float, suffix: str) -> str:
    return f"img_{index:06d}_{timestamp_ms:011.5f}_ms{suffix.lower()}"


def records_from_pose(pose7_wxyz: np.ndarray, timestamps_ms: np.ndarray) -> list[dict]:
    records = []
    for pose, timestamp in zip(pose7_wxyz, timestamps_ms):
        pose_list = [float(x) for x in pose]
        records.append(
            {
                "robot_time_stamps": float(timestamp),
                "ts_pose_fb": pose_list,
                "ts_pose_command": pose_list,
            }
        )
    return records


def records_from_wrench(wrench: np.ndarray, timestamps_ms: np.ndarray) -> list[dict]:
    records = []
    for value, timestamp in zip(wrench, timestamps_ms):
        records.append(
            {
                "wrench_time_stamps": float(timestamp),
                "wrench": [float(x) for x in value],
            }
        )
    return records


def write_json(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2)


def convert_session(
    session_dir: Path,
    output_root: Path,
    episode_name: str,
    camera_name: str = "",
) -> Path:
    session_dir = session_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    data = load_session(session_dir, camera_name)

    episode_dir = output_root / episode_name
    rgb_dir = episode_dir / "rgb_0"
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    rgb_dir.mkdir(parents=True)

    t0_s = common_start_time_s(data)
    camera_timestamps_ms = to_relative_ms(data.camera_timestamps_s, t0_s)
    robot_timestamps_ms = to_relative_ms(data.robot_timestamps_s, t0_s)
    wrench_timestamps_ms = to_relative_ms(data.wrench_timestamps_s, t0_s)

    for idx, (src, timestamp_ms) in enumerate(zip(data.color_files, camera_timestamps_ms)):
        dst = rgb_dir / image_name(idx, float(timestamp_ms), src.suffix)
        shutil.copy2(src, dst)

    write_json(
        episode_dir / "robot_data_0.json",
        records_from_pose(data.tcp_pose7_wxyz, robot_timestamps_ms),
    )
    write_json(
        episode_dir / "wrench_data_0.json",
        records_from_wrench(data.wrench, wrench_timestamps_ms),
    )

    metadata = {
        "source_session": str(data.session_dir),
        "source_camera": data.camera_dir.name,
        "timestamp_unit": "ms_relative_to_episode_start",
        "episode_start_time_host_s": t0_s,
        "pose_order": "[x, y, z, qw, qx, qy, qz]",
        "command_source": "ts_pose_fb_0 copied from recorded TCP feedback",
    }
    write_json(episode_dir / "conversion_metadata.json", [metadata])
    return episode_dir


def convert_sessions(args: argparse.Namespace) -> list[Path]:
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} exists. Re-run with --force to overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    sessions = discover_sessions(args.sessions)
    episode_dirs = []
    for idx, session_dir in enumerate(sessions):
        episode_name = f"{args.episode_prefix}_{idx:06d}"
        episode_dir = convert_session(
            session_dir=session_dir,
            output_root=output_root,
            episode_name=episode_name,
            camera_name=args.camera_name,
        )
        print(f"{episode_name}: {session_dir} -> {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def main() -> None:
    args = parse_args()
    episode_dirs = convert_sessions(args)
    print(f"Wrote ACP raw dataset: {args.output} ({len(episode_dirs)} episodes)")


if __name__ == "__main__":
    main()
