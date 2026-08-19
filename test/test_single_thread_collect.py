import sys
import threading
import types
import importlib
from pathlib import Path

import numpy as np
import pytest


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


def import_dual_collect(monkeypatch):
    monkeypatch.setitem(sys.modules, "flexivrdk", types.SimpleNamespace())
    sys.modules.pop("dual_collect", None)
    return importlib.import_module("dual_collect")


def test_acquire_cycle_uses_one_common_timestamp_and_one_robot_read():
    from single_thread_collect import acquire_cycle

    class Camera:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def get(self):
            self.calls += 1
            return (
                np.full((2, 3, 3), self.value, dtype=np.uint8),
                np.full((2, 3), self.value, dtype=np.uint16),
            )

    class StateReader:
        def __init__(self):
            self.calls = 0

        def read_robot_sample(self):
            self.calls += 1
            return (
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.arange(7, dtype=np.float64),
                np.arange(6, dtype=np.float64) + 10.0,
                np.arange(6, dtype=np.float64) + 20.0,
            )

    class Gripper:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            return 0.04

    tactile_frame = types.SimpleNamespace(
        left=types.SimpleNamespace(timestamp_host_s=200.0),
        right=types.SimpleNamespace(timestamp_host_s=201.0),
    )
    tactile_reader = types.SimpleNamespace(read_frame=lambda: tactile_frame)
    cameras = {"cam_main": Camera(1), "cam_wrist": Camera(2)}
    state_reader = StateReader()
    gripper = Gripper()

    sample = acquire_cycle(
        cycle_index=7,
        cycle_timestamp_host_s=123.456,
        scheduled_monotonic_ns=10_000_000,
        cameras=cameras,
        state_reader=state_reader,
        slave_gripper=gripper,
        tactile_reader=tactile_reader,
        use_gripper=True,
        host_clock=lambda: 300.0,
        monotonic_clock_ns=lambda: 20_000_000,
    )

    assert sample.cycle_index == 7
    assert sample.cycle_timestamp_host_s == 123.456
    assert state_reader.calls == 1
    assert gripper.calls == 1
    assert all(camera.calls == 1 for camera in cameras.values())
    np.testing.assert_allclose(sample.tcp_pose, [1, 2, 3, 0, 0, 0, 1, 0.04])
    np.testing.assert_allclose(sample.q, [0, 1, 2, 3, 4, 5, 6, 0.04])
    np.testing.assert_allclose(sample.ext_wrench_in_tcp, np.arange(6) + 20.0)
    assert sample.tactile_frame is tactile_frame


def test_collect_aligned_data_saves_equal_counts_and_identical_timestamps(
    tmp_path, monkeypatch
):
    import single_thread_collect as stc

    class ImmediateRateControl:
        rates = []

        def __init__(self, rate_hz):
            self.rates.append(rate_hz)

        def wait_next(self):
            return 10_000_000

    monkeypatch.setattr(stc, "AlignedRateControl", ImmediateRateControl)

    def image_writer(path, image):
        Path(path).write_bytes(np.asarray(image).tobytes())
        return True

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imwrite=image_writer))
    stop_event = threading.Event()

    class StateReader:
        def __init__(self):
            self.calls = 0

        def read_robot_sample(self):
            self.calls += 1
            value = float(self.calls)
            if self.calls == 2:
                stop_event.set()
            return (
                np.full(3, value),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.full(7, value),
                np.full(6, value),
                np.full(6, value + 10.0),
            )

    class Camera:
        def get(self):
            return np.ones((2, 3, 3), np.uint8), np.ones((2, 3), np.uint16)

    def fingertip(value):
        return types.SimpleNamespace(
            timestamp_host_s=500.0 + value,
            marker_offset=np.full((1, 2, 2), value, np.float32),
            force_torque=np.full(6, value, np.float64),
            force_norm=np.full((2, 3, 3), value, np.float32),
            rectify=np.full((2, 3, 3), value, np.uint8),
            difference=np.full((2, 3), value, np.uint8),
            depth=np.full((2, 3), value, np.uint16),
        )

    class TactileReader:
        def __init__(self):
            self.calls = 0

        def read_frame(self):
            self.calls += 1
            return types.SimpleNamespace(
                left=fingertip(float(self.calls)),
                right=fingertip(float(self.calls + 10)),
            )

    state_reader = StateReader()
    stc.collect_aligned_teleop_data(
        state_reader=state_reader,
        slave_gripper=None,
        cameras={"cam_test": Camera()},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        fps=10,
        use_gripper=False,
        tactile_reader=TactileReader(),
        status_period=0,
        writer_queue_size=2,
    )

    camera_ts = np.load(tmp_path / "cam_test" / "timestamps_host_s.npy")
    robot_ts = np.load(tmp_path / "robot" / "timestamps_host_s.npy")
    wrench_ts = np.load(tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy")
    left_ts = np.load(tmp_path / "tactile" / "left" / "timestamps_host_s.npy")
    right_ts = np.load(tmp_path / "tactile" / "right" / "timestamps_host_s.npy")
    write_ts = np.load(tmp_path / "timing" / "write_completed_host_s.npy")
    cycle_duration = np.load(tmp_path / "timing" / "cycle_duration_s.npy")
    deadline_overrun = np.load(tmp_path / "timing" / "deadline_overrun_s.npy")

    assert ImmediateRateControl.rates == [10]
    assert state_reader.calls == 2
    assert camera_ts.shape == (2,)
    np.testing.assert_array_equal(robot_ts, camera_ts)
    np.testing.assert_array_equal(wrench_ts, camera_ts)
    np.testing.assert_array_equal(left_ts, camera_ts)
    np.testing.assert_array_equal(right_ts, camera_ts)
    assert write_ts.shape == (2,)
    assert np.all(write_ts >= camera_ts)
    assert cycle_duration.shape == (2,)
    assert deadline_overrun.shape == (2,)
    assert np.all(cycle_duration >= 0.0)
    assert np.all(deadline_overrun >= 0.0)
    assert np.load(tmp_path / "robot" / "tcp_pose.npy").shape == (2, 8)
    assert np.load(tmp_path / "ext_wrench_in_tcp.npy").shape == (2, 6)
    assert len(list((tmp_path / "cam_test" / "color").glob("*.png"))) == 2
    assert len(list((tmp_path / "tactile" / "left" / "depth").glob("*.png"))) == 2


def test_dual_collect_utils_routes_collection_to_single_aligned_worker(
    tmp_path, monkeypatch
):
    import dual_collect_utils as dcu
    import single_thread_collect as stc

    captured = {}

    def record_call(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(stc, "collect_aligned_teleop_data", record_call)
    stop_event = threading.Event()

    dcu.collect_teleop_data(
        state_reader="state",
        slave_gripper="gripper",
        cameras={"cam": "camera"},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        fps=10,
        use_gripper=True,
        status_period=5,
        tactile_reader="tactile",
    )

    assert captured == {
        "state_reader": "state",
        "slave_gripper": "gripper",
        "cameras": {"cam": "camera"},
        "session_dir": str(tmp_path),
        "stop_event": stop_event,
        "fps": 10,
        "use_gripper": True,
        "status_period": 5,
        "tactile_reader": "tactile",
    }


def test_cli_defaults_to_10_hz_collection_with_30_fps_camera_device(
    monkeypatch,
):
    dual_collect = import_dual_collect(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dual_collect.py",
            "-1",
            "master",
            "-2",
            "slave",
            "--save-root",
            "data",
            "--use-gripper",
            "false",
        ],
    )

    args = dual_collect.parse_args()

    assert args.fps == 10
    assert args.camera_device_fps == 30
    assert args.camera_fps is None
    assert args.robot_fps is None
    assert args.force_fps is None
    assert args.tactile_fps is None


def test_metadata_declares_capture_time_alignment_and_write_timing(monkeypatch):
    dual_collect = import_dual_collect(monkeypatch)
    args = types.SimpleNamespace(
        fps=10,
        camera_device_fps=30,
        camera_fps=None,
        robot_fps=None,
        force_fps=None,
        tactile_fps=None,
        use_gripper=False,
        use_tactile=False,
        tactile_left_sensor_sn="left",
        tactile_right_sensor_sn="right",
        initial_gripper_width=0.0,
    )

    metadata = dual_collect.build_metadata(args, {"cam": "serial"}, "tdk", "saved")

    assert metadata["collection_mode"] == "single_rate_single_acquisition_thread"
    assert metadata["collection_fps"] == 10
    assert metadata["camera_device_fps"] == 30
    assert metadata["effective_camera_fps"] == 10
    assert metadata["effective_robot_fps"] == 10
    assert metadata["effective_force_fps"] == 10
    assert metadata["effective_tactile_fps"] == 10
    assert metadata["timestamp_semantics"] == (
        "one shared host timestamp assigned at the start of each acquisition cycle"
    )
    assert metadata["timing_stream_files"]["write_completed"] == (
        "timing/write_completed_host_s.npy"
    )


def test_launcher_uses_one_10_hz_collection_rate():
    launcher = (COLLECT_DIR / "run_dual_collect.sh").read_text(encoding="utf-8")

    assert 'FPS="10"' in launcher
    assert 'CAMERA_DEVICE_FPS="30"' in launcher
    assert "--camera-device-fps" in launcher
    assert "--camera-fps" not in launcher
    assert "--robot-fps" not in launcher
    assert "--force-fps" not in launcher
    assert "--tactile-fps" not in launcher


def test_writer_failure_discards_the_entire_aligned_cycle(tmp_path, monkeypatch):
    import single_thread_collect as stc

    class ImmediateRateControl:
        def __init__(self, _rate_hz):
            pass

        def wait_next(self):
            return 10_000_000

    monkeypatch.setattr(stc, "AlignedRateControl", ImmediateRateControl)
    writes = []

    def fail_second_write(path, image):
        Path(path).write_bytes(np.asarray(image).tobytes())
        writes.append(path)
        return len(writes) != 2

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(imwrite=fail_second_write),
    )
    stop_event = threading.Event()

    class StateReader:
        def read_robot_sample(self):
            stop_event.set()
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.zeros(6),
                np.zeros(6),
            )

    class Camera:
        def get(self):
            return np.ones((2, 3, 3), np.uint8), np.ones((2, 3), np.uint16)

    with pytest.raises(IOError, match="Failed to write aligned cycle image"):
        stc.collect_aligned_teleop_data(
            state_reader=StateReader(),
            slave_gripper=None,
            cameras={"cam_test": Camera()},
            session_dir=str(tmp_path),
            stop_event=stop_event,
            fps=10,
            use_gripper=False,
            status_period=0,
        )

    assert list(tmp_path.rglob("*.png")) == []
    assert np.load(tmp_path / "robot" / "tcp_pose.npy").shape == (0, 8)
    assert np.load(tmp_path / "ext_wrench_in_tcp.npy").shape == (0, 6)
    assert np.load(tmp_path / "cam_test" / "timestamps_host_s.npy").shape == (0,)
    assert np.load(tmp_path / "timing" / "write_completed_host_s.npy").shape == (0,)
