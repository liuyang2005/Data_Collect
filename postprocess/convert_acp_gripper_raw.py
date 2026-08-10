#!/usr/bin/env python3
"""Convert Data_Collect sessions to ACP raw episodes with gripper labels.

The recorded eighth TCP column is gripper width feedback.  ACP needs a command-like
training target, so this converter also derives a binary action with hysteresis:
0 means closed and 1 means open.  Width feedback is retained as an observation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import convert_acp_raw as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Data_Collect sessions to ACP raw data with gripper fields."
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="Session directories, or roots containing session directories.",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--camera-name", default="")
    parser.add_argument("--episode-prefix", default="episode")
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.065,
        help="Width at or below which the derived action becomes closed (0).",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.075,
        help="Width at or above which the derived action becomes open (1).",
    )
    parser.add_argument(
        "--require-tactile",
        action="store_true",
        help="Fail if tactile/force_torque.npy, marker_offset.npy, or timestamps are absent.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_gripper_width(session_dir: Path) -> np.ndarray:
    robot_path = session_dir / "robot" / "tcp_pose.npy"
    if not robot_path.is_file():
        robot_path = session_dir / "tcps.npy"
    tcps = np.asarray(np.load(robot_path), dtype=np.float64)
    if tcps.ndim != 2 or tcps.shape[1] != 8:
        raise ValueError(
            "TCP pose must have shape (T, 8): [x,y,z,qx,qy,qz,qw,width], "
            f"got {tcps.shape} from {robot_path}"
        )
    width = tcps[:, 7]
    if not np.all(np.isfinite(width)):
        raise ValueError(f"gripper width contains non-finite values: {robot_path}")
    return width


def derive_gripper_action(
    width: np.ndarray,
    close_threshold: float = 0.065,
    open_threshold: float = 0.075,
) -> np.ndarray:
    """Return a stable binary open/close label from measured width feedback."""
    width = np.asarray(width, dtype=np.float64).reshape(-1)
    if width.size == 0:
        raise ValueError("gripper width is empty")
    if not close_threshold < open_threshold:
        raise ValueError("gripper close threshold must be below open threshold")

    midpoint = 0.5 * (close_threshold + open_threshold)
    state = 1.0 if width[0] >= midpoint else 0.0
    action = np.empty_like(width)
    for index, value in enumerate(width):
        if value <= close_threshold:
            state = 0.0
        elif value >= open_threshold:
            state = 1.0
        action[index] = state
    return action


def add_gripper_fields(
    episode_dir: Path,
    width: np.ndarray,
    close_threshold: float,
    open_threshold: float,
) -> None:
    robot_path = episode_dir / "robot_data_0.json"
    records = json.loads(robot_path.read_text(encoding="utf-8"))
    if len(records) != len(width):
        raise ValueError(
            f"robot/gripper length mismatch in {episode_dir}: "
            f"{len(records)} vs {len(width)}"
        )
    action = derive_gripper_action(width, close_threshold, open_threshold)
    for record, width_value, action_value in zip(records, width, action):
        record["gripper_width_fb"] = float(width_value)
        record["gripper_action"] = float(action_value)
    base.write_json(robot_path, records)

    metadata_path = episode_dir / "conversion_metadata.json"
    metadata_records = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_records[0]
    metadata.update(
        {
            "gripper_width_source": "recorded TCP feedback column 7",
            "gripper_action_semantics": "binary: 0=closed, 1=open",
            "gripper_action_derivation": "width-feedback hysteresis",
            "gripper_close_threshold": float(close_threshold),
            "gripper_open_threshold": float(open_threshold),
        }
    )
    base.write_json(metadata_path, [metadata])


def add_tactile_fields(episode_dir: Path, session_dir: Path, required: bool) -> bool:
    tactile_dir = session_dir / "tactile"
    paths = {
        "force_torque": tactile_dir / "force_torque.npy",
        "marker_offset": tactile_dir / "marker_offset.npy",
        "timestamps": tactile_dir / "timestamps_host_s.npy",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        if required:
            raise FileNotFoundError("missing tactile files: " + ", ".join(missing))
        return False

    force_torque = np.asarray(np.load(paths["force_torque"]), dtype=np.float64)
    marker_offset = np.asarray(np.load(paths["marker_offset"]), dtype=np.float32)
    timestamps_s = np.asarray(np.load(paths["timestamps"]), dtype=np.float64).reshape(-1)
    if force_torque.ndim != 2 or force_torque.shape[1] != 6:
        raise ValueError(f"tactile force_torque must have shape (T, 6), got {force_torque.shape}")
    if marker_offset.ndim != 4 or marker_offset.shape[-1] != 2:
        raise ValueError(
            f"tactile marker_offset must have shape (T, H, W, 2), got {marker_offset.shape}"
        )
    if not (len(force_torque) == len(marker_offset) == len(timestamps_s)):
        raise ValueError(
            "tactile stream length mismatch: "
            f"force={len(force_torque)}, marker={len(marker_offset)}, time={len(timestamps_s)}"
        )

    metadata_path = episode_dir / "conversion_metadata.json"
    metadata_records = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_records[0]
    t0_s = float(metadata["episode_start_time_host_s"])
    timestamps_ms = base.to_relative_ms(timestamps_s, t0_s)
    np.save(episode_dir / "tactile_force_torque_0.npy", force_torque)
    np.save(episode_dir / "tactile_marker_offset_0.npy", marker_offset)
    np.save(episode_dir / "tactile_time_stamps_0.npy", timestamps_ms)
    metadata.update(
        {
            "tactile_force_torque_shape": list(force_torque.shape[1:]),
            "tactile_marker_offset_shape": list(marker_offset.shape[1:]),
            "tactile_timestamp_unit": "ms_relative_to_episode_start",
        }
    )
    base.write_json(metadata_path, [metadata])
    return True


def convert_session(
    session_dir: Path,
    output_root: Path,
    episode_name: str,
    camera_name: str = "",
    close_threshold: float = 0.065,
    open_threshold: float = 0.075,
    require_tactile: bool = False,
) -> Path:
    session_dir = session_dir.expanduser().resolve()
    width = load_gripper_width(session_dir)
    episode_dir = base.convert_session(
        session_dir=session_dir,
        output_root=output_root,
        episode_name=episode_name,
        camera_name=camera_name,
    )
    add_gripper_fields(episode_dir, width, close_threshold, open_threshold)
    add_tactile_fields(episode_dir, session_dir, required=require_tactile)
    return episode_dir


def convert_sessions(args: argparse.Namespace) -> list[Path]:
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} exists. Re-run with --force to overwrite."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    episode_dirs = []
    for index, session_dir in enumerate(base.discover_sessions(args.sessions)):
        episode_name = f"{args.episode_prefix}_{index:06d}"
        episode_dir = convert_session(
            session_dir=session_dir,
            output_root=output_root,
            episode_name=episode_name,
            camera_name=args.camera_name,
            close_threshold=args.gripper_close_threshold,
            open_threshold=args.gripper_open_threshold,
            require_tactile=args.require_tactile,
        )
        print(f"{episode_name}: {session_dir} -> {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def main() -> None:
    args = parse_args()
    episode_dirs = convert_sessions(args)
    print(f"Wrote ACP gripper raw dataset: {args.output} ({len(episode_dirs)} episodes)")


if __name__ == "__main__":
    main()
