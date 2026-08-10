#!/usr/bin/env python3
"""Convert sessions using inferred causal gripper-command labels.

The recorder stores slave gripper width feedback, not the master-side command.
Using thresholded feedback at the query time as both an observation and action
target lets a policy copy its current state and prevents it from initiating a
close. This converter anchors each state event at a reliable width threshold,
searches backwards for sustained motion onset, and compensates for latency.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import convert_acp_gripper_raw as legacy


@dataclass(frozen=True)
class CommandTransition:
    state: int
    threshold_crossing_index: int
    threshold_crossing_time_s: float
    motion_onset_index: int
    motion_onset_time_s: float
    command_index: int
    command_time_s: float
    lead_from_threshold_s: float
    width_at_motion_onset_m: float
    width_at_command_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--camera-name", default="")
    parser.add_argument("--episode-prefix", default="episode")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.065)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.075)
    parser.add_argument(
        "--motion-speed-threshold-m-s",
        type=float,
        default=0.002,
        help="Minimum directed feedback speed treated as gripper motion.",
    )
    parser.add_argument(
        "--motion-max-gap-s",
        type=float,
        default=0.08,
        help="Bridge feedback quantization gaps up to this duration.",
    )
    parser.add_argument(
        "--motion-search-window-s",
        type=float,
        default=0.75,
        help="Maximum backwards search from a threshold crossing.",
    )
    parser.add_argument(
        "--motion-min-delta-m",
        type=float,
        default=0.003,
        help="Minimum directed width change in a detected event.",
    )
    parser.add_argument(
        "--command-latency-s",
        type=float,
        default=0.05,
        help="Shift the command before the first detected feedback motion.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Odd median-filter window in robot samples.",
    )
    parser.add_argument("--require-tactile", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_robot_timestamps(session_dir: Path, expected_length: int) -> np.ndarray:
    path = session_dir / "robot" / "timestamps_host_s.npy"
    timestamps = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if timestamps.shape != (expected_length,):
        raise ValueError(
            f"robot timestamp/width length mismatch: {timestamps.shape} vs {(expected_length,)}"
        )
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"robot timestamps must be finite and strictly increasing: {path}")
    return timestamps


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 0 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if window == 1 or values.size == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=-1)


def _transition_indices(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    return np.flatnonzero(states[1:] != states[:-1]) + 1


def _find_motion_onset(
    smoothed_width: np.ndarray,
    timestamps_s: np.ndarray,
    crossing_index: int,
    new_state: int,
    speed_threshold_m_s: float,
    max_gap_s: float,
    search_window_s: float,
    min_delta_m: float,
) -> int:
    direction = -1.0 if new_state == 0 else 1.0
    directed_speed = direction * np.diff(smoothed_width) / np.diff(timestamps_s)
    speed_sample_indices = np.flatnonzero(directed_speed >= speed_threshold_m_s) + 1
    window_start_s = timestamps_s[crossing_index] - search_window_s
    candidates = speed_sample_indices[
        (speed_sample_indices <= crossing_index)
        & (timestamps_s[speed_sample_indices] >= window_start_s)
    ]
    if candidates.size == 0:
        raise ValueError(
            f"no directed gripper motion before transition at index {crossing_index}"
        )

    cluster_start = candidates.size - 1
    while cluster_start > 0:
        previous = candidates[cluster_start - 1]
        current = candidates[cluster_start]
        if timestamps_s[current] - timestamps_s[previous] > max_gap_s:
            break
        cluster_start -= 1
    onset = max(0, int(candidates[cluster_start]) - 1)
    directed_delta = direction * (
        smoothed_width[crossing_index] - smoothed_width[onset]
    )
    if directed_delta < min_delta_m:
        raise ValueError(
            "detected gripper motion is too small before transition at index "
            f"{crossing_index}: {directed_delta:.6f} m"
        )
    return onset


def derive_gripper_command(
    width: np.ndarray,
    timestamps_s: np.ndarray,
    close_threshold: float = 0.065,
    open_threshold: float = 0.075,
    speed_threshold_m_s: float = 0.002,
    max_gap_s: float = 0.08,
    search_window_s: float = 0.75,
    min_delta_m: float = 0.003,
    command_latency_s: float = 0.05,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[CommandTransition]]:
    """Infer a causal binary command sequence from width feedback motion."""
    width = np.asarray(width, dtype=np.float64).reshape(-1)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if width.size == 0 or timestamps_s.shape != width.shape:
        raise ValueError("width and timestamps must be nonempty one-dimensional arrays")
    if not np.all(np.isfinite(width)):
        raise ValueError("gripper width contains NaN or Inf")
    if not np.all(np.isfinite(timestamps_s)) or np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("gripper timestamps must be finite and strictly increasing")
    if not close_threshold < open_threshold:
        raise ValueError("gripper close threshold must be below open threshold")
    for name, value in {
        "speed_threshold_m_s": speed_threshold_m_s,
        "max_gap_s": max_gap_s,
        "search_window_s": search_window_s,
        "min_delta_m": min_delta_m,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(command_latency_s) or command_latency_s < 0.0:
        raise ValueError("command_latency_s must be finite and nonnegative")

    smoothed = median_smooth(width, smoothing_window)
    feedback_state = legacy.derive_gripper_action(
        smoothed,
        close_threshold=close_threshold,
        open_threshold=open_threshold,
    )
    command = np.full_like(feedback_state, feedback_state[0])
    transitions: list[CommandTransition] = []
    last_command_index = -1
    for crossing_index in _transition_indices(feedback_state):
        new_state = int(feedback_state[crossing_index])
        onset_index = _find_motion_onset(
            smoothed,
            timestamps_s,
            int(crossing_index),
            new_state,
            speed_threshold_m_s,
            max_gap_s,
            search_window_s,
            min_delta_m,
        )
        inferred_command_time_s = timestamps_s[onset_index] - command_latency_s
        command_index = int(np.searchsorted(timestamps_s, inferred_command_time_s, side="left"))
        command_index = max(last_command_index + 1, min(command_index, width.size - 1))
        if command_index >= crossing_index:
            raise ValueError(
                "inferred gripper command must precede its feedback threshold crossing: "
                f"command={command_index}, crossing={crossing_index}"
            )
        command[command_index:] = float(new_state)
        transitions.append(
            CommandTransition(
                state=new_state,
                threshold_crossing_index=int(crossing_index),
                threshold_crossing_time_s=float(timestamps_s[crossing_index]),
                motion_onset_index=onset_index,
                motion_onset_time_s=float(timestamps_s[onset_index]),
                command_index=command_index,
                command_time_s=float(timestamps_s[command_index]),
                lead_from_threshold_s=float(
                    timestamps_s[crossing_index] - timestamps_s[command_index]
                ),
                width_at_motion_onset_m=float(width[onset_index]),
                width_at_command_m=float(width[command_index]),
            )
        )
        last_command_index = command_index
    return command, feedback_state, transitions


def add_gripper_command_fields(
    episode_dir: Path,
    width: np.ndarray,
    timestamps_s: np.ndarray,
    args: argparse.Namespace,
) -> list[CommandTransition]:
    command, feedback_state, transitions = derive_gripper_command(
        width,
        timestamps_s,
        close_threshold=args.gripper_close_threshold,
        open_threshold=args.gripper_open_threshold,
        speed_threshold_m_s=args.motion_speed_threshold_m_s,
        max_gap_s=args.motion_max_gap_s,
        search_window_s=args.motion_search_window_s,
        min_delta_m=args.motion_min_delta_m,
        command_latency_s=args.command_latency_s,
        smoothing_window=args.smoothing_window,
    )
    robot_path = episode_dir / "robot_data_0.json"
    records = json.loads(robot_path.read_text(encoding="utf-8"))
    if len(records) != len(width):
        raise ValueError(
            f"robot/gripper length mismatch in {episode_dir}: {len(records)} vs {len(width)}"
        )
    for record, width_value, command_value, state_value in zip(
        records, width, command, feedback_state
    ):
        record["gripper_width_fb"] = float(width_value)
        record["gripper_feedback_state"] = float(state_value)
        record["gripper_action"] = float(command_value)
    legacy.base.write_json(robot_path, records)

    metadata_path = episode_dir / "conversion_metadata.json"
    metadata_records = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_records[0]
    metadata.update(
        {
            "gripper_width_source": "recorded TCP feedback column 7",
            "gripper_action_semantics": "binary command: 0=closed, 1=open",
            "gripper_action_derivation": "feedback motion onset with latency compensation",
            "gripper_close_threshold": float(args.gripper_close_threshold),
            "gripper_open_threshold": float(args.gripper_open_threshold),
            "gripper_motion_speed_threshold_m_s": float(args.motion_speed_threshold_m_s),
            "gripper_motion_max_gap_s": float(args.motion_max_gap_s),
            "gripper_motion_search_window_s": float(args.motion_search_window_s),
            "gripper_motion_min_delta_m": float(args.motion_min_delta_m),
            "gripper_command_latency_s": float(args.command_latency_s),
            "gripper_smoothing_window": int(args.smoothing_window),
            "gripper_command_transitions": [asdict(item) for item in transitions],
        }
    )
    legacy.base.write_json(metadata_path, [metadata])
    return transitions


def convert_session(
    session_dir: Path,
    output_root: Path,
    episode_name: str,
    args: argparse.Namespace,
) -> tuple[Path, list[CommandTransition]]:
    session_dir = session_dir.expanduser().resolve()
    width = legacy.load_gripper_width(session_dir)
    timestamps_s = load_robot_timestamps(session_dir, len(width))
    episode_dir = legacy.base.convert_session(
        session_dir=session_dir,
        output_root=output_root,
        episode_name=episode_name,
        camera_name=args.camera_name,
    )
    transitions = add_gripper_command_fields(episode_dir, width, timestamps_s, args)
    legacy.add_tactile_fields(episode_dir, session_dir, required=args.require_tactile)
    return episode_dir, transitions


def convert_sessions(args: argparse.Namespace) -> list[Path]:
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} exists. Re-run with --force to overwrite."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    episode_dirs: list[Path] = []
    all_transitions: list[CommandTransition] = []
    for index, session_dir in enumerate(legacy.base.discover_sessions(args.sessions)):
        episode_name = f"{args.episode_prefix}_{index:06d}"
        episode_dir, transitions = convert_session(
            session_dir, output_root, episode_name, args
        )
        transition_text = ", ".join(
            f"{'open' if item.state else 'close'} lead={item.lead_from_threshold_s:.3f}s"
            for item in transitions
        )
        print(f"{episode_name}: {session_dir} -> {episode_dir} | {transition_text}")
        episode_dirs.append(episode_dir)
        all_transitions.extend(transitions)

    if not all_transitions:
        raise ValueError("no gripper command transitions were detected")
    leads = [item.lead_from_threshold_s for item in all_transitions]
    summary = {
        "episodes": len(episode_dirs),
        "transitions": len(all_transitions),
        "close_transitions": sum(item.state == 0 for item in all_transitions),
        "open_transitions": sum(item.state == 1 for item in all_transitions),
        "lead_from_threshold_s": {
            "min": min(leads),
            "median": float(np.median(leads)),
            "max": max(leads),
        },
    }
    (output_root / "gripper_command_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return episode_dirs


def main() -> None:
    args = parse_args()
    episode_dirs = convert_sessions(args)
    print(
        f"Wrote ACP command-labelled gripper raw dataset: {args.output} "
        f"({len(episode_dirs)} episodes)"
    )


if __name__ == "__main__":
    main()
