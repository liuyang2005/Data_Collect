import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


POSTPROCESS_DIR = Path(__file__).resolve().parents[1] / "postprocess"
if str(POSTPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_DIR))


def import_converter():
    sys.modules.pop("convert_acp_gripper_command_raw", None)
    return importlib.import_module("convert_acp_gripper_command_raw")


def synthetic_motion() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(60, dtype=np.float64) * 0.01
    width = np.concatenate(
        [
            np.full(10, 0.080),
            np.linspace(0.080, 0.055, 16),
            np.full(12, 0.055),
            np.linspace(0.055, 0.080, 16),
            np.full(6, 0.080),
        ]
    )
    return width, timestamps


def converter_args() -> argparse.Namespace:
    return argparse.Namespace(
        gripper_close_threshold=0.065,
        gripper_open_threshold=0.075,
        motion_speed_threshold_m_s=0.002,
        motion_max_gap_s=0.08,
        motion_search_window_s=0.75,
        motion_min_delta_m=0.003,
        command_latency_s=0.05,
        smoothing_window=5,
    )


def test_command_transitions_precede_feedback_state_changes():
    converter = import_converter()
    width, timestamps = synthetic_motion()
    command, feedback_state, transitions = converter.derive_gripper_command(
        width, timestamps
    )

    assert [item.state for item in transitions] == [0, 1]
    assert np.count_nonzero(np.diff(command)) == 2
    assert transitions[0].command_index < transitions[0].threshold_crossing_index
    assert transitions[1].command_index < transitions[1].threshold_crossing_index
    assert feedback_state[transitions[0].command_index] == 1
    assert command[transitions[0].command_index] == 0
    assert feedback_state[transitions[1].command_index] == 0
    assert command[transitions[1].command_index] == 1
    assert transitions[0].width_at_command_m > 0.075
    assert transitions[1].width_at_command_m < 0.065


def test_non_increasing_timestamps_are_rejected():
    converter = import_converter()
    width, timestamps = synthetic_motion()
    timestamps[20] = timestamps[19]
    with pytest.raises(ValueError, match="strictly increasing"):
        converter.derive_gripper_command(width, timestamps)


def test_add_fields_keeps_feedback_separate_from_command(tmp_path):
    converter = import_converter()
    width, timestamps = synthetic_motion()
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    records = [{"robot_time_stamps": float(value)} for value in timestamps]
    (episode_dir / "robot_data_0.json").write_text(json.dumps(records))
    (episode_dir / "conversion_metadata.json").write_text(json.dumps([{}]))

    transitions = converter.add_gripper_command_fields(
        episode_dir, width, timestamps, converter_args()
    )

    converted = json.loads((episode_dir / "robot_data_0.json").read_text())
    close_index = transitions[0].command_index
    assert converted[close_index]["gripper_action"] == 0
    assert converted[close_index]["gripper_feedback_state"] == 1
    assert converted[close_index]["gripper_width_fb"] > 0.075
    metadata = json.loads(
        (episode_dir / "conversion_metadata.json").read_text()
    )[0]
    assert metadata["gripper_action_semantics"] == "binary command: 0=closed, 1=open"
    assert len(metadata["gripper_command_transitions"]) == 2
