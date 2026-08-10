import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


class OutputTypes:
    Marker2D = "marker"
    ForceResultant = "force"
    ForceNorm = "force_norm"
    Rectify = "rectify"
    Difference = "difference"
    Depth = "depth"


class FakeSensor:
    def __init__(self, baseline_value=1.0, frame_value=3.0):
        self.released = False
        self.baseline_marker = np.full(
            (1, 2, 2), baseline_value, dtype=np.float32
        )
        self.baseline_samples = None
        self.frame_marker = np.full((1, 2, 2), frame_value, dtype=np.float32)
        self.force_torque = np.arange(6, dtype=np.float64) + frame_value
        self.force_norm = np.full((2, 3, 3), frame_value, dtype=np.float32)
        self.rectify = np.full((2, 3, 3), frame_value, dtype=np.uint8)
        self.difference = np.full((2, 3), frame_value, dtype=np.uint8)
        self.depth = np.full((2, 3), frame_value, dtype=np.uint16)

    def selectSensorInfo(self, *outputs):
        if outputs == (OutputTypes.Marker2D,):
            if self.baseline_samples is not None:
                return next(self.baseline_samples).copy()
            return self.baseline_marker.copy()
        assert outputs == (
            OutputTypes.Marker2D,
            OutputTypes.ForceResultant,
            OutputTypes.ForceNorm,
            OutputTypes.Rectify,
            OutputTypes.Difference,
            OutputTypes.Depth,
        )
        return (
            self.frame_marker.copy(),
            self.force_torque.copy(),
            self.force_norm.copy(),
            self.rectify.copy(),
            self.difference.copy(),
            self.depth.copy(),
        )

    def release(self):
        self.released = True


def import_xense_tactile():
    sys.modules.pop("xense_tactile", None)
    return importlib.import_module("xense_tactile")


def make_reader(module, left, right, **kwargs):
    sensors = {"OG001453": left, "OG001455": right}
    kwargs.setdefault("baseline_duration_s", 0.0)

    def factory(serial_number, *, mac_addr):
        assert mac_addr == "gripper_8a429d6ea337"
        return sensors[serial_number]

    return module.XenseTactileReader(
        left_sensor_serial_number="OG001453",
        right_sensor_serial_number="OG001455",
        mac_addr="gripper_8a429d6ea337",
        sensor_factory=factory,
        output_types=OutputTypes,
        **kwargs,
    )


def test_reader_connects_both_sensors_and_returns_complete_frames():
    module = import_xense_tactile()
    left = FakeSensor(baseline_value=1.0, frame_value=3.0)
    right = FakeSensor(baseline_value=10.0, frame_value=14.0)
    calls = []

    def factory(serial_number, *, mac_addr):
        calls.append((serial_number, mac_addr))
        return {"OG001453": left, "OG001455": right}[serial_number]

    reader = module.XenseTactileReader(
        left_sensor_serial_number="OG001453",
        right_sensor_serial_number="OG001455",
        mac_addr="gripper_8a429d6ea337",
        sensor_factory=factory,
        output_types=OutputTypes,
        baseline_duration_s=0.0,
    )

    reader.connect()
    frame = reader.read_frame()

    assert calls == [
        ("OG001453", "gripper_8a429d6ea337"),
        ("OG001455", "gripper_8a429d6ea337"),
    ]
    assert reader.baseline_ready is True
    np.testing.assert_array_equal(
        frame.left.marker_offset,
        np.full(left.baseline_marker.shape, 2.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        frame.right.marker_offset,
        np.full(right.baseline_marker.shape, 4.0, dtype=np.float32),
    )
    for fingertip, sensor in ((frame.left, left), (frame.right, right)):
        np.testing.assert_array_equal(fingertip.force_torque, sensor.force_torque)
        np.testing.assert_array_equal(fingertip.force_norm, sensor.force_norm)
        np.testing.assert_array_equal(fingertip.rectify, sensor.rectify)
        np.testing.assert_array_equal(fingertip.difference, sensor.difference)
        np.testing.assert_array_equal(fingertip.depth, sensor.depth)
        assert fingertip.timestamp_host_s > 0.0
        assert np.isfinite(fingertip.timestamp_host_s)

    reader.close()

    assert reader.baseline_ready is False
    assert left.released is True
    assert right.released is True


def test_reader_uses_independent_median_marker_baselines():
    module = import_xense_tactile()
    left = FakeSensor(frame_value=8.0)
    right = FakeSensor(frame_value=18.0)
    left.baseline_samples = iter(
        np.full((1, 2, 2), value, dtype=np.float32)
        for value in (10.0, 2.0, 6.0)
    )
    right.baseline_samples = iter(
        np.full((1, 2, 2), value, dtype=np.float32)
        for value in (20.0, 12.0, 16.0)
    )
    reader = make_reader(
        module,
        left,
        right,
        baseline_duration_s=3.0 / 60.0,
        baseline_rate_hz=60.0,
    )

    reader.connect()
    frame = reader.read_frame()

    np.testing.assert_array_equal(frame.left.marker_offset, left.frame_marker - 6.0)
    np.testing.assert_array_equal(
        frame.right.marker_offset, right.frame_marker - 16.0
    )


def test_reader_rejects_duplicate_sensor_serial_numbers():
    module = import_xense_tactile()

    with pytest.raises(ValueError, match="must be different"):
        module.XenseTactileReader(
            left_sensor_serial_number="OG001453",
            right_sensor_serial_number="OG001453",
            mac_addr="gripper_8a429d6ea337",
        )


def test_reader_rejects_force_resultant_without_six_components():
    module = import_xense_tactile()
    left = FakeSensor()
    left.force_torque = np.arange(5, dtype=np.float64)
    reader = make_reader(module, left, FakeSensor())

    with pytest.raises(RuntimeError, match="left Xense ForceResultant.*6 components"):
        reader.read_frame()


def test_reader_rejects_invalid_force_norm_shape():
    module = import_xense_tactile()
    right = FakeSensor()
    right.force_norm = np.zeros((2, 3), dtype=np.float32)
    reader = make_reader(module, FakeSensor(), right)

    with pytest.raises(RuntimeError, match=r"right Xense ForceNorm.*\(H, W, 3\)"):
        reader.read_frame()


def test_reader_releases_created_sensor_when_second_creation_fails():
    module = import_xense_tactile()
    left = FakeSensor()

    def factory(serial_number, *, mac_addr):
        if serial_number == "OG001453":
            return left
        raise RuntimeError("right create failed")

    reader = module.XenseTactileReader(
        left_sensor_serial_number="OG001453",
        right_sensor_serial_number="OG001455",
        mac_addr="gripper_8a429d6ea337",
        sensor_factory=factory,
        output_types=OutputTypes,
        baseline_duration_s=0.0,
    )

    with pytest.raises(RuntimeError, match="right create failed"):
        reader.connect()

    assert left.released is True
    assert reader.baseline_ready is False


def test_reader_releases_both_sensors_when_baseline_is_invalid():
    module = import_xense_tactile()
    left = FakeSensor()
    right = FakeSensor()
    right.baseline_marker = np.zeros((3,), dtype=np.float32)
    reader = make_reader(module, left, right)

    with pytest.raises(RuntimeError, match="must end in an x/y dimension"):
        reader.connect()

    assert left.released is True
    assert right.released is True
    assert reader.baseline_ready is False
