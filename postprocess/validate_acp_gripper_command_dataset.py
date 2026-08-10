#!/usr/bin/env python3
"""Validate an episode-grouped ACP dataset with inferred gripper commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ACP_ROOT = WORKSPACE_ROOT / "adaptive_compliance_policy_extend"
sys.path.insert(0, str(ACP_ROOT))

from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs  # noqa: E402

register_codecs()
import zarr  # noqa: E402


ROBOT_FIELDS = {
    "ts_pose_fb_0": (7,),
    "ts_pose_command_0": (7,),
    "ts_pose_virtual_target_0": (7,),
    "stiffness_0": (),
    "gripper_width_0": (1,),
    "gripper_action_0": (1,),
    "robot_time_stamps_0": (),
}
MODALITY_FIELDS = {
    "rgb_0": ("episode_rgb0_len", (480, 640, 3)),
    "rgb_time_stamps_0": ("episode_rgb0_len", ()),
    "wrench_0": ("episode_wrench0_len", (6,)),
    "wrench_filtered_0": ("episode_wrench0_len", (6,)),
    "wrench_time_stamps_0": ("episode_wrench0_len", ()),
    "tactile_force_torque_0": ("episode_tactile0_len", (6,)),
    "tactile_marker_features_0": ("episode_tactile0_len", (6,)),
    "tactile_time_stamps_0": ("episode_tactile0_len", ()),
}
TIMESTAMP_FIELDS = (
    "robot_time_stamps_0",
    "rgb_time_stamps_0",
    "wrench_time_stamps_0",
    "tactile_time_stamps_0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--close-threshold", type=float, default=0.065)
    parser.add_argument("--open-threshold", type=float, default=0.075)
    return parser.parse_args()


def require_shape(name: str, shape: tuple[int, ...], length: int, tail: tuple[int, ...]) -> None:
    expected = (length, *tail)
    if shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {shape}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    root = zarr.open(str(dataset_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("dataset must contain data and meta groups")

    data = root["data"]
    meta = root["meta"]
    episode_names = sorted(data.group_keys())
    if not episode_names:
        raise ValueError("dataset contains no episodes")

    metadata = {}
    for key in (
        "episode_robot0_len",
        "episode_rgb0_len",
        "episode_wrench0_len",
        "episode_tactile0_len",
    ):
        if key not in meta:
            raise ValueError(f"missing metadata array: {key}")
        values = np.asarray(meta[key], dtype=np.int64)
        if values.shape != (len(episode_names),):
            raise ValueError(
                f"{key}: expected {len(episode_names)} entries, got {values.shape}"
            )
        metadata[key] = values

    command_widths = {"close": [], "open": []}
    threshold_leads_s = {"close": [], "open": []}
    total_robot_samples = 0

    for episode_index, episode_name in enumerate(episode_names):
        episode = data[episode_name]
        robot_length = int(metadata["episode_robot0_len"][episode_index])
        total_robot_samples += robot_length

        for field, tail_shape in ROBOT_FIELDS.items():
            if field not in episode:
                raise ValueError(f"{episode_name}: missing {field}")
            require_shape(field, episode[field].shape, robot_length, tail_shape)

        for field, (length_key, tail_shape) in MODALITY_FIELDS.items():
            if field not in episode:
                raise ValueError(f"{episode_name}: missing {field}")
            expected_length = int(metadata[length_key][episode_index])
            require_shape(field, episode[field].shape, expected_length, tail_shape)

        for field in episode.array_keys():
            if field == "rgb_0":
                rgb = episode[field]
                if rgb.dtype != np.dtype("uint8"):
                    raise ValueError(f"{episode_name}: rgb_0 must be uint8")
                # Decode both ends to verify that the JPEG-XL codec and chunks are readable.
                _ = np.asarray(rgb[0])
                _ = np.asarray(rgb[-1])
                continue
            values = np.asarray(episode[field])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{episode_name}: {field} contains NaN or Inf")

        for field in TIMESTAMP_FIELDS:
            timestamps = np.asarray(episode[field], dtype=np.float64).reshape(-1)
            if np.any(np.diff(timestamps) <= 0.0):
                raise ValueError(f"{episode_name}: {field} is not strictly increasing")

        action = np.asarray(episode["gripper_action_0"], dtype=np.float64).reshape(-1)
        width = np.asarray(episode["gripper_width_0"], dtype=np.float64).reshape(-1)
        timestamps = np.asarray(
            episode["robot_time_stamps_0"], dtype=np.float64
        ).reshape(-1)
        if not np.all(np.isin(action, (0.0, 1.0))):
            raise ValueError(f"{episode_name}: gripper action is not binary")
        changes = np.flatnonzero(action[1:] != action[:-1]) + 1
        transition_states = action[changes].astype(np.int64).tolist()
        if action[0] != 1.0 or transition_states != [0, 1]:
            raise ValueError(
                f"{episode_name}: expected open -> close -> open, got "
                f"initial={action[0]:g}, transitions={transition_states}"
            )

        close_index, open_index = (int(value) for value in changes)
        command_widths["close"].append(float(width[close_index]))
        command_widths["open"].append(float(width[open_index]))

        close_candidates = np.flatnonzero(
            width[close_index:open_index] <= args.close_threshold
        )
        open_candidates = np.flatnonzero(width[open_index:] >= args.open_threshold)
        if close_candidates.size == 0 or open_candidates.size == 0:
            raise ValueError(f"{episode_name}: command has no matching feedback crossing")
        close_crossing = close_index + int(close_candidates[0])
        open_crossing = open_index + int(open_candidates[0])
        if close_crossing <= close_index or open_crossing <= open_index:
            raise ValueError(f"{episode_name}: command does not lead feedback")
        # ACP raw/zarr timestamps are stored in milliseconds.
        threshold_leads_s["close"].append(
            float(timestamps[close_crossing] - timestamps[close_index]) / 1000.0
        )
        threshold_leads_s["open"].append(
            float(timestamps[open_crossing] - timestamps[open_index]) / 1000.0
        )

    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "max": float(np.max(array)),
        }

    report = {
        "dataset": str(dataset_path),
        "episodes": len(episode_names),
        "robot_samples": total_robot_samples,
        "transition_sequence_per_episode": [0, 1],
        "gripper_action_semantics": "0=closed, 1=open",
        "width_at_command_m": {
            key: stats(values) for key, values in command_widths.items()
        },
        "command_lead_to_feedback_threshold_s": {
            key: stats(values) for key, values in threshold_leads_s.items()
        },
        "status": "PASS",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
