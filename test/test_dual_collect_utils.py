import importlib
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest


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
            np.array([idx, idx + 0.01, idx + 0.02, idx + 0.03, idx + 0.04, idx + 0.05]),
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


def test_init_xense_uses_direct_sdk_and_preserves_legacy_gripper_interface(monkeypatch):
    dcu = import_dual_collect_utils()
    calls = {}

    class FakeBackend:
        def get_gripper_status(self):
            return {"position": 25.0}

        def set_position(self, position_mm, velocity_mm_s, force_n):
            calls["set_position"] = (position_mm, velocity_mm_s, force_n)

        def release(self):
            calls["released"] = True

    backend = FakeBackend()

    class FakeXenseGripper:
        @staticmethod
        def create(*, mac_addr):
            calls["mac_addr"] = mac_addr
            return backend

    monkeypatch.setitem(
        sys.modules,
        "xensegripper",
        types.SimpleNamespace(XenseGripper=FakeXenseGripper),
    )

    gripper = dcu.init_xense("d254505bfaaa", "slave_xense")

    assert calls["mac_addr"] == "d254505bfaaa"
    assert gripper.read() == pytest.approx(0.025)
    gripper.move(0.04)
    gripper.close()
    assert calls["set_position"] == (40.0, 80.0, 20.0)
    assert calls["released"] is True


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


def test_default_d405_profile_uses_uniform_640x480_resolution(monkeypatch):
    dcu = import_dual_collect_utils()
    install_fake_pyrealsense2(monkeypatch)

    profile = dcu.CAMERA_PROFILES[f"cam_{dcu.WRIST_CAMERA_SERIAL}_wrist"]
    assert profile == {
        "serial": dcu.WRIST_CAMERA_SERIAL,
        "model": "D405",
        "width": 640,
        "height": 480,
    }

    camera = dcu.RealSenseD415(
        serial=profile["serial"],
        fps=30,
        name=f"cam_{dcu.WRIST_CAMERA_SERIAL}_wrist",
        model=profile["model"],
        width=profile["width"],
        height=profile["height"],
    )

    assert camera.model == "D405"
    assert camera.config.streams == [
        ("depth", 640, 480, "z16", 30),
        ("color", 640, 480, "bgr8", 30),
    ]


def test_init_cameras_closes_open_camera_when_next_profile_fails(monkeypatch):
    dcu = import_dual_collect_utils()
    opened = []

    class FakeCamera:
        def __init__(self, *, serial, **_kwargs):
            if serial == "bad-camera":
                raise RuntimeError("camera startup failed")
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(dcu, "RealSenseD415", FakeCamera)
    profiles = {
        "cam_good": {"serial": "good-camera"},
        "cam_bad": {"serial": "bad-camera"},
    }

    with pytest.raises(RuntimeError, match="camera startup failed"):
        dcu.init_cameras(profiles, fps=30)

    assert len(opened) == 1
    assert opened[0].closed is True


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


def test_dual_collect_defaults_to_no_feedback_and_new_left_tactile_sensor(
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
            "--slave-gripper-id",
            "d254505bfaaa",
            "--save-root",
            "data",
        ],
    )

    args = dual_collect.parse_args()
    metadata = dual_collect.build_metadata(args, {}, "tdk", "saved")

    assert args.wrench_feedback_scale == 0.0
    assert args.tactile_sensor_sn == "OG000451"
    assert metadata["wrench_feedback_scale"] == 0.0
    assert metadata["tcp_pose_source"] == (
        "TransparentCartesianTeleopLAN.instances(0)[1].states().tcp_pose"
    )


def test_dual_collect_rejects_non_comparison_feedback_scale(monkeypatch, capsys):
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
            "d254505bfaaa",
            "--save-root",
            "data",
            "--wrench-feedback-scale",
            "0.5",
        ],
    )

    with pytest.raises(SystemExit):
        dual_collect.parse_args()

    assert "invalid choice" in capsys.readouterr().err


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


def test_dual_collect_accepts_single_finger_tactile_settings(monkeypatch):
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
            "--use-tactile",
            "true",
            "--tactile-fps",
            "60",
            "--tactile-sensor-sn",
            "OG000451",
            "--tactile-mac-addr",
            "d254505bfaaa",
        ],
    )

    args = dual_collect.parse_args()

    assert args.use_tactile is True
    assert args.tactile_fps == 60
    assert args.tactile_sensor_sn == "OG000451"
    assert args.tactile_mac_addr == "d254505bfaaa"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--tactile-fps", "0", "--tactile-mac-addr", "d254505bfaaa"],
        ["--tactile-fps", "60", "--tactile-mac-addr", ""],
        [
            "--tactile-fps",
            "60",
            "--tactile-mac-addr",
            "d254505bfaaa",
            "--tactile-sensor-sn",
            "",
        ],
    ],
)
def test_dual_collect_rejects_invalid_enabled_tactile_settings(
    monkeypatch, extra_args
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
            "--slave-gripper-id",
            "gripper",
            "--save-root",
            "data",
            "--use-tactile",
            "true",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit):
        dual_collect.parse_args()


def test_build_metadata_describes_tactile_stream(monkeypatch):
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
            "d254505bfaaa",
            "--save-root",
            "data",
            "--use-tactile",
            "true",
            "--tactile-fps",
            "75",
            "--tactile-sensor-sn",
            "OG000451",
            "--tactile-mac-addr",
            "d254505bfaaa",
        ],
    )
    args = dual_collect.parse_args()

    metadata = dual_collect.build_metadata(args, {}, "tdk", "saved")

    assert metadata["effective_tactile_fps"] == 75
    assert metadata["tactile_source"] == "xensesdk.Sensor.OutputType"
    assert metadata["tactile_stream_files"] == {
        "marker_offset": "tactile/marker_offset.npy",
        "force_torque": "tactile/force_torque.npy",
        "timestamps": "tactile/timestamps_host_s.npy",
        "rectify": "tactile/rectify/*.png",
        "difference": "tactile/difference/*.png",
        "depth": "tactile/depth/*.png",
    }


def test_start_recording_passes_tactile_reader_and_independent_fps(
    tmp_path, monkeypatch
):
    dual_collect = import_dual_collect(monkeypatch)
    dcu = import_dual_collect_utils()
    tactile_reader = object()
    captured = {}

    class RecordingThread:
        def __init__(self, *, target, kwargs, daemon):
            captured["target"] = target
            captured["kwargs"] = kwargs
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(dual_collect.threading, "Thread", RecordingThread)
    monkeypatch.setattr(dual_collect, "build_metadata", lambda *_args: {})
    monkeypatch.setattr(dcu, "create_session_dirs", lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(dcu, "write_metadata", lambda *_args, **_kwargs: None)
    args = types.SimpleNamespace(
        session_name=None,
        save_root=str(tmp_path),
        fps=30,
        use_gripper=False,
        camera_fps=15,
        robot_fps=100,
        force_fps=200,
        tactile_fps=75,
    )

    _, _, collect_thread = dual_collect.start_recording(
        args=args,
        state_reader=object(),
        slave_gripper=None,
        tactile_reader=tactile_reader,
        cameras={},
        d415_cameras={},
        tdk_tcp_pose_order="tdk",
        saved_tcp_pose_order="saved",
    )

    assert collect_thread is not None
    assert captured["started"] is True
    assert captured["target"] is not dcu.collect_teleop_data
    assert captured["kwargs"]["worker"] is dcu.collect_teleop_data
    assert captured["kwargs"]["tactile_reader"] is tactile_reader
    assert captured["kwargs"]["tactile_fps"] == 75


def test_main_connects_passes_and_closes_tactile_reader(monkeypatch):
    dual_collect = import_dual_collect(monkeypatch)
    dcu = import_dual_collect_utils()
    instances = []
    captured = {}

    class FakeClosable:
        def __init__(self):
            self.closed = False
            self.moves = []

        def move(self, width):
            self.moves.append(width)

        def close(self):
            self.closed = True

    master_gripper = FakeClosable()
    slave_gripper = FakeClosable()
    camera = FakeClosable()

    class FakeTactileReader:
        def __init__(self, sensor_serial_number, mac_addr):
            self.sensor_serial_number = sensor_serial_number
            self.mac_addr = mac_addr
            self.connected = False
            self.closed = False
            instances.append(self)

        def connect(self):
            self.connected = True

        def close(self):
            self.closed = True

    tactile_module = types.ModuleType("xense_tactile")
    tactile_module.XenseTactileReader = FakeTactileReader
    monkeypatch.setitem(sys.modules, "xense_tactile", tactile_module)

    class FakeTeleopPair:
        def __init__(self, *_args, **_kwargs):
            self.wrench_feedback_scale = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def set_wrench_feedback_scale(self, factor):
            self.wrench_feedback_scale = factor
            captured["wrench_feedback_scale"] = factor

    transparent_module = types.ModuleType("transparent_teleop")
    transparent_module.SAVED_TCP_POSE_ORDER = "saved"
    transparent_module.TDK_TCP_POSE_ORDER = "tdk"
    transparent_module.TransparentCartesianTeleopPair = FakeTeleopPair
    transparent_module.TeleopSlaveStateReader = lambda pair: ("state", pair)
    monkeypatch.setitem(sys.modules, "transparent_teleop", transparent_module)
    monkeypatch.setattr(
        dcu,
        "init_cameras",
        lambda *_args, **_kwargs: {"cam_test": camera},
    )
    monkeypatch.setattr(dcu, "init_xense", lambda *_args, **_kwargs: slave_gripper)
    monkeypatch.setattr(
        dcu,
        "init_angler_controller",
        lambda *_args, **_kwargs: master_gripper,
    )
    monkeypatch.setattr(
        dual_collect,
        "run_keyboard_loop",
        lambda *args: captured.setdefault("tactile_reader", args[6]),
    )
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
            "true",
            "--slave-gripper-id",
            "d254505bfaaa",
            "--use-tactile",
            "true",
            "--tactile-sensor-sn",
            "OG000451",
            "--tactile-mac-addr",
            "d254505bfaaa",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        dual_collect.main()

    assert exit_info.value.code == 0
    assert len(instances) == 1
    reader = instances[0]
    assert reader.sensor_serial_number == "OG000451"
    assert reader.mac_addr == "d254505bfaaa"
    assert reader.connected is True
    assert captured["wrench_feedback_scale"] == 0.0
    assert captured["tactile_reader"] is reader
    assert reader.closed is True
    assert len(slave_gripper.moves) == 2
    assert master_gripper.closed is True
    assert slave_gripper.closed is True
    assert camera.closed is True


def test_read_robot_sample_rejects_invalid_tcp_velocity_shape():
    dcu = import_dual_collect_utils()

    class InvalidVelocityReader:
        def read_robot_sample(self):
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.zeros(5),
                np.zeros(6),
            )

    with pytest.raises(ValueError, match="tcp_vel must have shape \\(6,\\)"):
        dcu.read_robot_sample(InvalidVelocityReader())


def test_collect_teleop_data_writes_aligned_robot_arrays(tmp_path, monkeypatch):
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

    tcp_pose = np.load(tmp_path / "robot" / "tcp_pose.npy")
    tcp_vel = np.load(tmp_path / "robot" / "tcp_vel.npy")
    q = np.load(tmp_path / "robot" / "q.npy")
    timestamps = np.load(tmp_path / "robot" / "timestamps_host_s.npy")

    assert tcp_pose.shape == (3, 8)
    assert tcp_vel.shape == (3, 6)
    assert q.shape == (3, 8)
    np.testing.assert_allclose(tcp_pose[:, -1], 0.0)
    np.testing.assert_allclose(q[:, -1], 0.0)
    np.testing.assert_allclose(tcp_vel[0], [1.0, 1.01, 1.02, 1.03, 1.04, 1.05])

    assert timestamps.shape == (3,)
    assert np.all(np.diff(timestamps) >= 0.0)

    assert not (tmp_path / "tcps.npy").exists()
    assert not (tmp_path / "angles.npy").exists()
    assert not (tmp_path / "tcp_vel.npy").exists()


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

    timestamps = np.load(tmp_path / "robot" / "timestamps_host_s.npy")
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

    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    np.save(robot_dir / "tcp_pose.npy", np.zeros((3, 8)))
    np.save(robot_dir / "tcp_vel.npy", np.zeros((3, 6)))
    np.save(robot_dir / "q.npy", np.zeros((3, 8)))
    np.save(tmp_path / "ext_wrench_in_tcp.npy", np.zeros((4, 6)))
    tactile_dir = tmp_path / "tactile"
    tactile_dir.mkdir()
    np.save(tactile_dir / "force_torque.npy", np.zeros((2, 6)))

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
        "robot": {"tcp_pose": 3, "tcp_vel": 3, "q": 3},
        "force": 4,
        "tactile": 2,
    }
    assert "Saving episode" in caplog.text
    assert "Episode saved" in caplog.text
    assert "color=2" in caplog.text
    assert "tcp_pose=3" in caplog.text
    assert "tcp_vel=3" in caplog.text
    assert "force=4" in caplog.text
    assert "tactile=2" in caplog.text


def test_stop_collection_propagates_background_collection_failure(
    tmp_path, monkeypatch
):
    dual_collect = import_dual_collect(monkeypatch)
    dcu = import_dual_collect_utils()

    def fail_collection(**_kwargs):
        raise RuntimeError("background collection failed")

    monkeypatch.setattr(dcu, "collect_teleop_data", fail_collection)
    monkeypatch.setattr(dual_collect, "build_metadata", lambda *_args: {})
    monkeypatch.setattr(dcu, "create_session_dirs", lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(dcu, "write_metadata", lambda *_args, **_kwargs: None)
    args = types.SimpleNamespace(
        session_name=None,
        save_root=str(tmp_path),
        fps=30,
        use_gripper=False,
        camera_fps=15,
        robot_fps=100,
        force_fps=200,
        tactile_fps=60,
    )

    session_dir, stop_event, collect_thread = dual_collect.start_recording(
        args=args,
        state_reader=object(),
        slave_gripper=None,
        tactile_reader=object(),
        cameras={},
        d415_cameras={},
        tdk_tcp_pose_order="tdk",
        saved_tcp_pose_order="saved",
    )

    with pytest.raises(RuntimeError, match="background collection failed"):
        dual_collect.stop_collection(
            stop_event,
            collect_thread,
            session_dir=session_dir,
            camera_names=(),
        )


def test_collect_teleop_data_uses_independent_camera_robot_force_and_tactile_fps(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    tactile_module = importlib.import_module("xense_tactile")
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

    class TactileReader:
        def read_frame(self):
            time.sleep(0.001)
            return tactile_module.XenseTactileFrame(
                marker_offset=np.zeros((1, 2, 2), dtype=np.float32),
                force_torque=np.zeros(6, dtype=np.float64),
                rectify=np.zeros((2, 2, 3), dtype=np.uint8),
                difference=np.zeros((2, 2), dtype=np.uint8),
                depth=np.zeros((2, 2), dtype=np.uint16),
            )

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
        tactile_reader=TactileReader(),
        tactile_fps=60,
        use_gripper=False,
        status_period=0,
    )

    assert 5 in RecordingFastRateControl.rates
    assert 20 in RecordingFastRateControl.rates
    assert 80 in RecordingFastRateControl.rates
    assert 60 in RecordingFastRateControl.rates
    assert np.load(tmp_path / "robot" / "tcp_pose.npy").shape == (3, 8)
    assert np.load(tmp_path / "robot" / "tcp_vel.npy").shape == (3, 6)
    assert np.load(tmp_path / "robot" / "q.npy").shape == (3, 8)
    assert (tmp_path / "ext_wrench_in_tcp.npy").exists()
    assert (tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy").exists()
    assert (cam_dir / "timestamps_host_s.npy").exists()
    assert (tmp_path / "tactile" / "force_torque.npy").exists()
    assert (tmp_path / "tactile" / "timestamps_host_s.npy").exists()


def test_collect_tactile_stream_writes_aligned_arrays_and_image_triplets(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    tactile_module = importlib.import_module("xense_tactile")
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    def fake_imwrite(path, image):
        Path(path).write_bytes(np.asarray(image).tobytes())
        return True

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imwrite=fake_imwrite))
    stop_event = threading.Event()

    class TwoFrameReader:
        def __init__(self):
            self.count = 0

        def read_frame(self):
            self.count += 1
            value = self.count
            if self.count == 2:
                stop_event.set()
            return tactile_module.XenseTactileFrame(
                marker_offset=np.full((1, 2, 2), value, dtype=np.float32),
                force_torque=np.arange(6, dtype=np.float64) + value,
                rectify=np.full((2, 3, 3), value, dtype=np.uint8),
                difference=np.full((2, 3), value, dtype=np.uint8),
                depth=np.full((2, 3), value, dtype=np.uint16),
            )

    dcu.collect_tactile_stream(
        tactile_reader=TwoFrameReader(),
        session_dir=str(tmp_path),
        stop_event=stop_event,
        tactile_fps=60,
        status_period=0,
    )

    tactile_dir = tmp_path / "tactile"
    marker = np.load(tactile_dir / "marker_offset.npy")
    force = np.load(tactile_dir / "force_torque.npy")
    timestamps = np.load(tactile_dir / "timestamps_host_s.npy")
    assert marker.shape == (2, 1, 2, 2)
    assert marker.dtype == np.float32
    assert force.shape == (2, 6)
    assert force.dtype == np.float64
    assert timestamps.shape == (2,)
    for index in range(2):
        filename = f"{index:06d}.png"
        assert (tactile_dir / "rectify" / filename).exists()
        assert (tactile_dir / "difference" / filename).exists()
        assert (tactile_dir / "depth" / filename).exists()


def test_collect_tactile_stream_discards_incomplete_image_triplet(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    tactile_module = importlib.import_module("xense_tactile")
    monkeypatch.setattr(dcu, "RateControl", FastRateControl)

    def fail_mid_triplet(path, image):
        Path(path).write_bytes(np.asarray(image).tobytes())
        return "difference" not in Path(path).parts

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(imwrite=fail_mid_triplet),
    )
    stop_event = threading.Event()

    class OneFrameReader:
        def read_frame(self):
            stop_event.set()
            return tactile_module.XenseTactileFrame(
                marker_offset=np.ones((1, 2, 2), dtype=np.float32),
                force_torque=np.arange(6, dtype=np.float64),
                rectify=np.ones((2, 3, 3), dtype=np.uint8),
                difference=np.ones((2, 3), dtype=np.uint8),
                depth=np.ones((2, 3), dtype=np.uint16),
            )

    with pytest.raises(IOError, match="Failed to write tactile image"):
        dcu.collect_tactile_stream(
            tactile_reader=OneFrameReader(),
            session_dir=str(tmp_path),
            stop_event=stop_event,
            tactile_fps=60,
            status_period=0,
        )

    tactile_dir = tmp_path / "tactile"
    assert np.load(tactile_dir / "marker_offset.npy").shape[0] == 0
    assert np.load(tactile_dir / "force_torque.npy").shape == (0, 6)
    assert np.load(tactile_dir / "timestamps_host_s.npy").shape == (0,)
    assert list(tactile_dir.rglob("*.png")) == []


def test_collect_tactile_stream_propagates_missing_image_writer_dependency(
    tmp_path, monkeypatch
):
    dcu = import_dual_collect_utils()
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(ModuleNotFoundError, match="cv2"):
        dcu.collect_tactile_stream(
            tactile_reader=object(),
            session_dir=str(tmp_path),
            stop_event=threading.Event(),
            tactile_fps=60,
            status_period=0,
        )


def test_collect_teleop_data_propagates_tactile_reader_failure(tmp_path):
    dcu = import_dual_collect_utils()
    stop_event = threading.Event()
    tactile_called = threading.Event()

    class FailingTactileReader:
        def read_frame(self):
            tactile_called.set()
            raise RuntimeError("tactile failed")

    class BlockingStateReader:
        def read_robot_sample(self):
            assert tactile_called.wait(timeout=1.0)
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7),
                np.zeros(6),
                np.zeros(6),
            )

    with pytest.raises(RuntimeError, match="tactile failed"):
        dcu.collect_teleop_data(
            state_reader=BlockingStateReader(),
            slave_gripper=None,
            cameras={},
            session_dir=str(tmp_path),
            stop_event=stop_event,
            fps=1000,
            use_gripper=False,
            status_period=0,
            tactile_reader=FailingTactileReader(),
            tactile_fps=60,
        )

    assert stop_event.is_set()
    assert (tmp_path / "tactile" / "force_torque.npy").exists()


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
                np.zeros(6),
                np.array([idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5]),
            )

    dcu.collect_force_stream(
        state_reader=ForceStateReader(),
        session_dir=str(tmp_path),
        stop_event=force_stop_event,
        force_fps=80,
        status_period=0,
    )

    assert np.load(tmp_path / "robot" / "tcp_pose.npy").shape == (3, 8)
    assert np.load(tmp_path / "robot" / "tcp_vel.npy").shape == (3, 6)
    assert np.load(tmp_path / "robot" / "q.npy").shape == (3, 8)
    assert np.load(tmp_path / "ext_wrench_in_tcp.npy").shape == (5, 6)
    assert np.load(tmp_path / "ext_wrench_in_tcp_timestamps_host_s.npy").shape == (5,)
