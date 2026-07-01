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
    ext_wrench = np.load(tmp_path / "ext_wrench_in_tcp.npy")
    tcp_timestamps = np.load(tmp_path / "tcps_timestamps_host_s.npy")
    angle_timestamps = np.load(tmp_path / "angles_timestamps_host_s.npy")
    wrench_timestamps = np.load(tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy")

    assert tcps.shape == (3, 8)
    assert angles.shape == (3, 8)
    assert ext_wrench.shape == (3, 6)
    np.testing.assert_allclose(tcps[:, -1], 0.0)
    np.testing.assert_allclose(angles[:, -1], 0.0)
    np.testing.assert_allclose(ext_wrench[0], [1, 11, 21, 31, 41, 51])
    np.testing.assert_allclose(ext_wrench[2], [3, 13, 23, 33, 43, 53])

    assert tcp_timestamps.shape == (3,)
    np.testing.assert_allclose(angle_timestamps, tcp_timestamps)
    np.testing.assert_allclose(wrench_timestamps, tcp_timestamps)
    assert np.all(np.diff(tcp_timestamps) >= 0.0)

    assert not (tmp_path / "tcps").exists()
    assert not (tmp_path / "angles").exists()
