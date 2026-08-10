import importlib
import json
import sys
from pathlib import Path

import numpy as np


POSTPROCESS_DIR = Path(__file__).resolve().parents[1] / "postprocess"
if str(POSTPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_DIR))


def import_converter():
    sys.modules.pop("convert_acp_gripper_raw", None)
    return importlib.import_module("convert_acp_gripper_raw")


def make_session(session_dir: Path) -> None:
    robot_dir = session_dir / "robot"
    camera_dir = session_dir / "cam_wrist"
    color_dir = camera_dir / "color"
    robot_dir.mkdir(parents=True)
    color_dir.mkdir(parents=True)

    widths = [0.080, 0.070, 0.064, 0.069, 0.076]
    tcps = np.zeros((len(widths), 8), dtype=np.float64)
    tcps[:, 6] = 1.0
    tcps[:, 7] = widths
    np.save(robot_dir / "tcp_pose.npy", tcps)
    np.save(robot_dir / "tcp_vel.npy", np.zeros((len(widths), 6)))
    np.save(robot_dir / "q.npy", np.zeros((len(widths), 8)))
    np.save(robot_dir / "timestamps_host_s.npy", 10.0 + np.arange(len(widths)) * 0.01)

    np.save(session_dir / "ext_wrench_in_tcp.npy", np.zeros((len(widths), 6)))
    np.save(
        session_dir / "ext_wrench_in_tcp_timestamps_host_s.npy",
        10.0 + np.arange(len(widths)) * 0.01,
    )
    np.save(camera_dir / "timestamps_host_s.npy", 10.0 + np.arange(2) * 0.03)
    for index in range(2):
        (color_dir / f"{index:06d}.png").write_bytes(b"fake-image")

    tactile_dir = session_dir / "tactile"
    tactile_dir.mkdir()
    np.save(tactile_dir / "force_torque.npy", np.ones((3, 6), dtype=np.float64))
    np.save(tactile_dir / "marker_offset.npy", np.ones((3, 2, 2, 2), dtype=np.float32))
    np.save(tactile_dir / "timestamps_host_s.npy", 10.0 + np.arange(3) * 0.02)


def test_hysteresis_keeps_previous_state_inside_dead_band():
    converter = import_converter()
    action = converter.derive_gripper_action(
        np.array([0.080, 0.070, 0.064, 0.069, 0.076]),
        close_threshold=0.065,
        open_threshold=0.075,
    )
    np.testing.assert_array_equal(action, [1.0, 1.0, 0.0, 0.0, 1.0])


def test_convert_session_preserves_width_and_adds_binary_action(tmp_path):
    converter = import_converter()
    session_dir = tmp_path / "record_test"
    make_session(session_dir)

    episode_dir = converter.convert_session(
        session_dir,
        tmp_path / "raw",
        "episode_000000",
        camera_name="cam_wrist",
        require_tactile=True,
    )
    records = json.loads((episode_dir / "robot_data_0.json").read_text())
    np.testing.assert_allclose(
        [record["gripper_width_fb"] for record in records],
        [0.080, 0.070, 0.064, 0.069, 0.076],
    )
    assert [record["gripper_action"] for record in records] == [1, 1, 0, 0, 1]

    metadata = json.loads((episode_dir / "conversion_metadata.json").read_text())[0]
    assert metadata["gripper_action_semantics"] == "binary: 0=closed, 1=open"
    assert metadata["gripper_close_threshold"] == 0.065
    assert metadata["tactile_marker_offset_shape"] == [2, 2, 2]
    np.testing.assert_allclose(
        np.load(episode_dir / "tactile_time_stamps_0.npy"), [0.0, 20.0, 40.0]
    )
    assert np.load(episode_dir / "tactile_force_torque_0.npy").shape == (3, 6)
