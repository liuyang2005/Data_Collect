import importlib
import sys
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def import_datacollect_io():
    sys.modules.pop("datacollect_io", None)
    return importlib.import_module("datacollect_io")


def import_foar_converter():
    sys.modules.pop("convert_datacollect_to_foar", None)
    return importlib.import_module("convert_datacollect_to_foar")


def write_wrench_stream(session: Path) -> None:
    np.save(session / "ext_wrench_in_tcp.npy", np.arange(18, dtype=np.float64).reshape(3, 6))
    np.save(
        session / "ext_wrench_in_tcp_timestamps_host_s.npy",
        np.array([9.99, 10.01, 10.03], dtype=np.float64),
    )


def test_session_arrays_loads_new_robot_directory_with_shared_timestamps(tmp_path):
    io = import_datacollect_io()
    session = tmp_path / "record_new"
    robot = session / "robot"
    robot.mkdir(parents=True)
    tcp_pose = np.arange(16, dtype=np.float64).reshape(2, 8)
    tcp_vel = np.arange(12, dtype=np.float64).reshape(2, 6)
    q = np.arange(16, 32, dtype=np.float64).reshape(2, 8)
    robot_ts = np.array([10.0, 10.02], dtype=np.float64)
    np.save(robot / "tcp_pose.npy", tcp_pose)
    np.save(robot / "tcp_vel.npy", tcp_vel)
    np.save(robot / "q.npy", q)
    np.save(robot / "timestamps_host_s.npy", robot_ts)
    write_wrench_stream(session)

    arrays = io.session_arrays(session)

    assert io.is_session_dir(session)
    np.testing.assert_allclose(arrays["tcps"], tcp_pose)
    np.testing.assert_allclose(arrays["tcp_vel"], tcp_vel)
    np.testing.assert_allclose(arrays["angles"], q)
    np.testing.assert_allclose(arrays["tcp_ts"], robot_ts)
    assert arrays["angle_ts"] is arrays["tcp_ts"]
    assert arrays["wrench"].shape == (3, 6)
    assert arrays["wrench_ts"].shape == (3,)


def test_session_arrays_keeps_legacy_root_layout_compatible(tmp_path):
    io = import_datacollect_io()
    session = tmp_path / "record_legacy"
    session.mkdir()
    np.save(session / "tcps.npy", np.zeros((2, 8), dtype=np.float64))
    np.save(session / "tcps_timestamps_host_s.npy", np.array([10.0, 10.02]))
    np.save(session / "angles.npy", np.ones((2, 8), dtype=np.float64))
    np.save(session / "angles_timestamps_host_s.npy", np.array([10.0, 10.02]))
    write_wrench_stream(session)

    arrays = io.session_arrays(session)

    assert io.is_session_dir(session)
    assert arrays["tcps"].shape == (2, 8)
    assert arrays["angles"].shape == (2, 8)
    assert arrays["tcp_vel"] is None


def test_foar_converter_reads_new_robot_directory(tmp_path):
    converter = import_foar_converter()
    session = tmp_path / "record_new"
    robot = session / "robot"
    camera = session / "cam_test"
    (camera / "color").mkdir(parents=True)
    (camera / "depth").mkdir()
    robot.mkdir()
    for idx in range(2):
        (camera / "color" / f"{idx}.png").write_bytes(b"color")
        (camera / "depth" / f"{idx}.png").write_bytes(b"depth")
    np.save(camera / "timestamps_host_s.npy", np.array([10.0, 10.02]))
    np.save(
        robot / "tcp_pose.npy",
        np.array(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.08],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.04],
            ]
        ),
    )
    np.save(robot / "tcp_vel.npy", np.zeros((2, 6)))
    np.save(robot / "q.npy", np.column_stack([np.ones((2, 7)), [0.08, 0.04]]))
    np.save(robot / "timestamps_host_s.npy", np.array([10.0, 10.02]))
    write_wrench_stream(session)

    result = converter.convert_session(
        session=session,
        split="train",
        output_root=tmp_path / "foar",
        source_camera_name="cam_test",
        target_camera_id="test",
        calib_name="calib",
        image_mode="copy",
        wrench_frame="foar_raw",
    )

    high_freq = np.load(
        tmp_path
        / "foar"
        / "train"
        / "record_new"
        / "high_freq_data"
        / "force_torque_tcp_joint_timestamp.npy"
    )
    assert result["episode"] == "record_new"
    assert high_freq.shape == (3, 21)
