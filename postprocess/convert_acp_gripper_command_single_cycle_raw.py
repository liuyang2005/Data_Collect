#!/usr/bin/env python3
"""Build the pin-insertion command-label dataset from single-cycle episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import convert_acp_gripper_command_raw as command_converter


def canonical_sessions(
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict[str, object]]]:
    accepted: list[Path] = []
    skipped: list[dict[str, object]] = []
    for session_dir in command_converter.legacy.base.discover_sessions(args.sessions):
        resolved = session_dir.expanduser().resolve()
        width = command_converter.legacy.load_gripper_width(resolved)
        timestamps = command_converter.load_robot_timestamps(resolved, len(width))
        _, _, transitions = command_converter.derive_gripper_command(
            width,
            timestamps,
            close_threshold=args.gripper_close_threshold,
            open_threshold=args.gripper_open_threshold,
            speed_threshold_m_s=args.motion_speed_threshold_m_s,
            max_gap_s=args.motion_max_gap_s,
            search_window_s=args.motion_search_window_s,
            min_delta_m=args.motion_min_delta_m,
            command_latency_s=args.command_latency_s,
            smoothing_window=args.smoothing_window,
        )
        sequence = [item.state for item in transitions]
        if sequence == [0, 1]:
            accepted.append(resolved)
        else:
            skipped.append(
                {"session": str(resolved), "transition_sequence": sequence}
            )
    return accepted, skipped


def main() -> None:
    args = command_converter.parse_args()
    discovered_count = len(
        command_converter.legacy.base.discover_sessions(args.sessions)
    )
    accepted, skipped = canonical_sessions(args)
    if not accepted:
        raise ValueError("no single-cycle gripper episodes were found")
    for item in skipped:
        print(
            f"SKIP: {item['session']} has noncanonical gripper sequence "
            f"{item['transition_sequence']}"
        )

    filtered_args = argparse.Namespace(**vars(args))
    filtered_args.sessions = accepted
    episode_dirs = command_converter.convert_sessions(filtered_args)

    summary_path = args.output.expanduser().resolve() / "gripper_command_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "source_discovered_sessions": discovered_count,
            "single_cycle_required": True,
            "skipped_sessions": skipped,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote single-cycle ACP command dataset: {args.output} "
        f"({len(episode_dirs)} episodes, {len(skipped)} skipped)"
    )


if __name__ == "__main__":
    main()
