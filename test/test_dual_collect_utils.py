import importlib
import sys
import threading
import types
from pathlib import Path

import numpy as np


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


def import_dual_collect_utils():
    sys.modules.pop("dual_collect_utils", None)
    return importlib.import_module("dual_collect_utils")


def import_dual_collect(monkeypatch):
    monkeypatch.setitem(sys.modules, "flexivrdk", types.SimpleNamespace())
    sys.modules.pop("dual_collect", None)
    return importlib.import_module("dual_collect")


class FastRateControl:
    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.count = 0

    def sleep(self):
        self.count += 1
        return float(self.rate_hz)


class RecordingFastRateControl:
    rates = []

    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.count = 0
        RecordingFastRateControl.rates.append(rate_hz)

    def sleep(self):
        self.count += 1
        return float(self.rate_hz)


class FakeStateReader:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.count = 0

    def read_robot_sample(self):
        self.count += 1
        idx = float(self.count)
        if self.count >= 3:
            self.stop_event.set()
        return (
            np.array([idx, idx + 0.1, idx + 0.2]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5, idx + 6]),
            np.array([idx, idx + 10, idx + 20, idx + 30, idx + 40, idx + 50]),
        )


class FakeVideoProfile:
    def get_intrinsics(self):
        return types.SimpleNamespace(ppx=1.0, ppy=2.0, fx=3.0, fy=4.0)


class FakeProfile:
    def as_video_stream_profile(self):
        return FakeVideoProfile()

    def get_extrinsics_to(self, other):
        return types.SimpleNamespace(
            rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation=[0.1, 0.2, 0.3],
        )


class FakeFrame:
    def __init__(self, data):
        self.data = data

    def as_video_frame(self):
        return self

    def as_depth_frame(self):
        return self

    def get_profile(self):
        return FakeProfile()

    def get_data(self):
        return self.data

    def __bool__(self):
        return True


class FakeFrames:
    def __init__(self):
        self.color = FakeFrame(np.full((2, 2, 3), 7, dtype=np.uint8))
        self.depth = FakeFrame(np.full((2, 2), 11, dtype=np.uint16))

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class FakeDepthSensor:
    def get_depth_scale(self):
        return 0.001


class FakeDevice:
    def __init__(self):
        self.reset_called = False

    def first_depth_sensor(self):
        return FakeDepthSensor()

    def hardware_reset(self):
        self.reset_called = True


class FakePipelineProfile:
    def __init__(self, device):
        self.device = device

    def get_device(self):
        return self.device


class FakePipeline:
    def __init__(self, device):
        self.device = device
        self.started = False
        self.stopped = False

    def start(self, config):
        self.started = True
        return FakePipelineProfile(self.device)

    def stop(self):
        self.stopped = True

    def wait_for_frames(self):
        return FakeFrames()


class FakeConfig:
    def __init__(self):
        self.enabled_device = None
        self.streams = []

    def enable_device(self, serial):
        self.enabled_device = serial

    def enable_stream(self, *stream):
        self.streams.append(stream)


class FakeAlign:
    def __init__(self, stream):
        self.stream = stream
        self.called = False

    def process(self, frames):
        self.called = True
        return frames


def install_fake_pyrealsense2(monkeypatch):
    fake_device = FakeDevice()
    state = {}
    fake_rs = types.SimpleNamespace(
        stream=types.SimpleNamespace(depth="depth", color="color"),
        format=types.SimpleNamespace(z16="z16", bgr8="bgr8"),
        pipeline=lambda: state.setdefault("pipeline", FakePipeline(fake_device)),
        config=FakeConfig,
        align=lambda stream: state.setdefault("align", FakeAlign(stream)),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    return state


def test_init_xense_uses_r3kit_wrapper_and_nonblocking_mode(monkeypatch):
    dcu = import_dual_collect_utils()
    calls = {}

    class FakeXense:
        def __init__(self, id, name):
            calls["id"] = id
            calls["name"] = name
            self.blocking = None

        def block(self, blocking):
            calls["blocking"] = blocking
            self.blocking = blocking

    class UnexpectedXenseGripper:
        @staticmethod
        def create(*args, **kwargs):
            raise AssertionError("init_xense should use the r3kit Xense wrapper")

    for module_name in [
        "r3kit",
        "r3kit.devices",
        "r3kit.devices.gripper",
        "r3kit.devices.gripper.xense",
    ]:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    r3kit_xense_module = types.ModuleType("r3kit.devices.gripper.xense.xense")
    r3kit_xense_module.Xense = FakeXense
    monkeypatch.setitem(
        sys.modules,
        "r3kit.devices.gripper.xense.xense",
        r3kit_xense_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "xensegripper",
        types.SimpleNamespace(XenseGripper=UnexpectedXenseGripper),
    )

    gripper = dcu.init_xense("1659f0e0dde0", "slave_xense")

    assert isinstance(gripper, FakeXense)
    assert calls == {
        "id": "1659f0e0dde0",
        "name": "slave_xense",
        "blocking": False,
    }
    assert gripper.blocking is False


def test_realsense_d415_matches_r3kit_core_frame_alignment_and_calibration(monkeypatch):
    dcu = import_dual_collect_utils()
    state = install_fake_pyrealsense2(monkeypatch)

    camera = dcu.RealSenseD415(serial="cam123", fps=15, name="test_cam")
    color, depth = camera.get()

    assert state["pipeline"].started
    assert state["align"].called
    assert camera.depth_scale == 0.001
    np.testing.assert_allclose(camera.color_intrinsics, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(camera.depth2color[:3, 3], [0.1, 0.2, 0.3])
    assert color.dtype == np.uint8
    assert depth.dtype == np.uint16


def test_dual_collect_keeps_gripper_enabled_by_default(monkeypatch):
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
            "--slave-gripper-id",
            "gripper",
            "--save-root",
            "data",
        ],
    )

    args = dual_collect.parse_args()

    assert args.use_gripper is True


def test_dual_collect_accepts_independent_stream_fps(monkeypatch):
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
            "--slave-gripper-id",
            "gripper",
            "--save-root",
            "data",
            "--camera-fps",
            "15",
            "--robot-fps",
            "100",
            "--force-fps",
            "200",
        ],
    )

    args = dual_collect.parse_args()

    assert args.camera_fps == 15
    assert args.robot_fps == 100
    assert args.force_fps == 200


def test_collect_teleop_data_writes_one_npy_per_robot_stream(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    stop_event = threading.Event()
    state_reader = FakeStateReader(stop_event)

    dcu.collect_teleop_data(
        state_reader=state_reader,
        slave_gripper=None,
        cameras={},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        fps=1000,
        use_gripper=False,
        status_period=0,
    )

    tcps = np.load(tmp_path / "tcps.npy")
    angles = np.load(tmp_path / "angles.npy")
    tcp_timestamps = np.load(tmp_path / "tcps_timestamps_host_s.npy")
    angle_timestamps = np.load(tmp_path / "angles_timestamps_host_s.npy")

    assert tcps.shape == (3, 8)
    assert angles.shape == (3, 8)
    np.testing.assert_allclose(tcps[:, -1], 0.0)
    np.testing.assert_allclose(angles[:, -1], 0.0)

    assert tcp_timestamps.shape == (3,)
    np.testing.assert_allclose(angle_timestamps, tcp_timestamps)
    assert np.all(np.diff(tcp_timestamps) >= 0.0)

    assert not (tmp_path / "tcps").exists()
    assert not (tmp_path / "angles").exists()


def test_collect_camera_stream_writes_frames_and_host_timestamps(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    RecordingFastRateControl.rates = []
    monkeypatch.setattr(dcu, "RateControl", RecordingFastRateControl)

    writes = []
    fake_cv2 = types.SimpleNamespace(
        imwrite=lambda path, image: writes.append((path, image.shape)) or True,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    stop_event = threading.Event()

    class FakeCamera:
        def __init__(self):
            self.count = 0

        def get(self):
            self.count += 1
            if self.count >= 3:
                stop_event.set()
            return (
                np.full((2, 2, 3), self.count, dtype=np.uint8),
                np.full((2, 2), self.count, dtype=np.uint16),
            )

    cam_dir = tmp_path / "cam_test"
    (cam_dir / "color").mkdir(parents=True)
    (cam_dir / "depth").mkdir(parents=True)

    dcu.collect_camera_stream(
        cameras={"cam_test": FakeCamera()},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        camera_fps=7,
        status_period=0,
    )

    assert RecordingFastRateControl.rates == [7]
    assert len(writes) == 6
    timestamps = np.load(cam_dir / "timestamps_host_s.npy")
    assert timestamps.shape == (3,)
    assert np.all(np.diff(timestamps) >= 0.0)


def test_camera_collection_continues_when_image_writes_are_blocked(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    first_write_started = threading.Event()
    release_writes = threading.Event()
    writes = []

    def blocking_imwrite(path, image):
        first_write_started.set()
        release_writes.wait(timeout=1.0)
        writes.append((path, image.shape))
        return True

    fake_cv2 = types.SimpleNamespace(imwrite=blocking_imwrite)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    stop_event = threading.Event()

    class FakeCamera:
        def __init__(self):
            self.count = 0

        def get(self):
            self.count += 1
            if self.count >= 3:
                stop_event.set()
            return (
                np.full((2, 2, 3), self.count, dtype=np.uint8),
                np.full((2, 2), self.count, dtype=np.uint16),
            )

    camera = FakeCamera()
    cam_dir = tmp_path / "cam_test"
    (cam_dir / "color").mkdir(parents=True)
    (cam_dir / "depth").mkdir(parents=True)

    thread = threading.Thread(
        target=dcu.collect_camera_stream,
        kwargs={
            "cameras": {"cam_test": camera},
            "session_dir": str(tmp_path),
            "stop_event": stop_event,
            "camera_fps": 30,
            "status_period": 0,
        },
    )
    thread.start()

    assert first_write_started.wait(timeout=1.0)
    assert camera.count >= 3

    release_writes.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(writes) == 6


def test_camera_timestamp_is_recorded_after_frame_read(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    clock = {"value": 10.0}
    monkeypatch.setattr(dcu.time, "time", lambda: clock["value"])
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(imwrite=lambda path, image: True),
    )

    stop_event = threading.Event()

    class TimestampCamera:
        def get(self):
            clock["value"] = 20.0
            stop_event.set()
            return (
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint16),
            )

    cam_dir = tmp_path / "cam_test"
    (cam_dir / "color").mkdir(parents=True)
    (cam_dir / "depth").mkdir(parents=True)

    dcu.collect_camera_stream(
        cameras={"cam_test": TimestampCamera()},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        camera_fps=30,
        status_period=0,
    )

    timestamps = np.load(cam_dir / "timestamps_host_s.npy")
    np.testing.assert_allclose(timestamps, [20.0])


def test_robot_timestamp_is_recorded_after_state_read(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    clock = {"value": 10.0}
    monkeypatch.setattr(dcu.time, "time", lambda: clock["value"])
    stop_event = threading.Event()

    class TimestampStateReader:
        def read_robot_sample(self):
            clock["value"] = 20.0
            stop_event.set()
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.zeros(6),
            )

    dcu.collect_robot_stream(
        state_reader=TimestampStateReader(),
        slave_gripper=None,
        session_dir=str(tmp_path),
        stop_event=stop_event,
        robot_fps=100,
        use_gripper=False,
        status_period=0,
    )

    timestamps = np.load(tmp_path / "tcps_timestamps_host_s.npy")
    np.testing.assert_allclose(timestamps, [20.0])


def test_force_timestamp_is_recorded_after_state_read(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    clock = {"value": 10.0}
    monkeypatch.setattr(dcu.time, "time", lambda: clock["value"])
    stop_event = threading.Event()

    class TimestampStateReader:
        def read_robot_sample(self):
            clock["value"] = 20.0
            stop_event.set()
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.zeros(6),
            )

    dcu.collect_force_stream(
        state_reader=TimestampStateReader(),
        session_dir=str(tmp_path),
        stop_event=stop_event,
        force_fps=200,
        status_period=0,
    )

    timestamps = np.load(tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy")
    np.testing.assert_allclose(timestamps, [20.0])


def test_stop_collection_waits_for_completion_and_reports_final_counts(
    tmp_path, monkeypatch, caplog
):
    dual_collect = import_dual_collect(monkeypatch)

    cam_dir = tmp_path / "cam_test"
    (cam_dir / "color").mkdir(parents=True)
    (cam_dir / "depth").mkdir(parents=True)
    for idx in range(2):
        (cam_dir / "color" / f"{idx:016d}.png").touch()
        (cam_dir / "depth" / f"{idx:016d}.png").touch()

    np.save(tmp_path / "tcps.npy", np.zeros((3, 8)))
    np.save(tmp_path / "angles.npy", np.zeros((3, 8)))
    np.save(tmp_path / "ext_wrench_in_tcp.npy", np.zeros((4, 6)))

    class RecordingThread:
        def __init__(self):
            self.join_timeout = "not-called"

        def join(self, timeout=None):
            self.join_timeout = timeout

    collect_thread = RecordingThread()
    stop_event = threading.Event()

    with caplog.at_level("INFO"):
        summary = dual_collect.stop_collection(
            stop_event,
            collect_thread,
            session_dir=str(tmp_path),
            camera_names=["cam_test"],
        )

    assert stop_event.is_set()
    assert collect_thread.join_timeout is None
    assert summary == {
        "cameras": {"cam_test": {"color": 2, "depth": 2}},
        "robot": {"tcps": 3, "angles": 3},
        "force": 4,
    }
    assert "Saving episode" in caplog.text
    assert "Episode saved" in caplog.text
    assert "color=2" in caplog.text
    assert "tcps=3" in caplog.text
    assert "force=4" in caplog.text


def test_collect_teleop_data_uses_independent_camera_robot_and_force_fps(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    RecordingFastRateControl.rates = []
    monkeypatch.setattr(dcu, "RateControl", RecordingFastRateControl)

    fake_cv2 = types.SimpleNamespace(imwrite=lambda path, image: True)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class OneShotCamera:
        def get(self):
            return (
                np.full((2, 2, 3), 1, dtype=np.uint8),
                np.full((2, 2), 1, dtype=np.uint16),
            )

    cam_dir = tmp_path / "cam_test"
    (cam_dir / "color").mkdir(parents=True)
    (cam_dir / "depth").mkdir(parents=True)
    stop_event = threading.Event()
    state_reader = FakeStateReader(stop_event)

    dcu.collect_teleop_data(
        state_reader=state_reader,
        slave_gripper=None,
        cameras={"cam_test": OneShotCamera()},
        session_dir=str(tmp_path),
        stop_event=stop_event,
        fps=30,
        camera_fps=5,
        robot_fps=20,
        force_fps=80,
        use_gripper=False,
        status_period=0,
    )

    assert 5 in RecordingFastRateControl.rates
    assert 20 in RecordingFastRateControl.rates
    assert 80 in RecordingFastRateControl.rates
    assert np.load(tmp_path / "tcps.npy").shape == (3, 8)
    assert (tmp_path / "ext_wrench_in_tcp.npy").exists()
    assert (tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy").exists()
    assert (cam_dir / "timestamps_host_s.npy").exists()


def test_robot_and_force_streams_can_have_different_lengths(tmp_path, monkeypatch):
    dcu = import_dual_collect_utils()
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    robot_stop_event = threading.Event()
    robot_reader = FakeStateReader(robot_stop_event)

    dcu.collect_robot_stream(
        state_reader=robot_reader,
        slave_gripper=None,
        session_dir=str(tmp_path),
        stop_event=robot_stop_event,
        robot_fps=20,
        use_gripper=False,
        status_period=0,
    )

    force_stop_event = threading.Event()

    class ForceStateReader:
        def __init__(self):
            self.count = 0

        def read_robot_sample(self):
            self.count += 1
            idx = float(self.count)
            if self.count >= 5:
                force_stop_event.set()
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.array([idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5]),
            )

    dcu.collect_force_stream(
        state_reader=ForceStateReader(),
        session_dir=str(tmp_path),
        stop_event=force_stop_event,
        force_fps=80,
        status_period=0,
    )

    assert np.load(tmp_path / "tcps.npy").shape == (3, 8)
    assert np.load(tmp_path / "angles.npy").shape == (3, 8)
    assert np.load(tmp_path / "ext_wrench_in_tcp.npy").shape == (5, 6)
    assert np.load(tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy").shape == (5,)
