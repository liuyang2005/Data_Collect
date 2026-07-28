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
    Rectify = "rectify"
    Difference = "difference"
    Depth = "depth"


class FakeSensor:
    def __init__(self):
        self.released = False
        self.baseline_marker = np.array(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=np.float32,
        )
        self.frame_marker = self.baseline_marker + 2.0
        self.force_torque = np.arange(6, dtype=np.float64)
        self.rectify = np.full((2, 3, 3), 10, dtype=np.uint8)
        self.difference = np.full((2, 3), 20, dtype=np.uint8)
        self.depth = np.full((2, 3), 30, dtype=np.uint16)

    def selectSensorInfo(self, *outputs):
        if outputs == (OutputTypes.Marker2D,):
            return self.baseline_marker.copy()
        assert outputs == (
            OutputTypes.Marker2D,
            OutputTypes.ForceResultant,
            OutputTypes.Rectify,
            OutputTypes.Difference,
            OutputTypes.Depth,
        )
        return (
            self.frame_marker.copy(),
            self.force_torque.copy(),
            self.rectify.copy(),
            self.difference.copy(),
            self.depth.copy(),
        )

    def release(self):
        self.released = True


def import_xense_tactile():
    sys.modules.pop("xense_tactile", None)
    return importlib.import_module("xense_tactile")


def test_reader_connects_configured_sensor_and_returns_complete_frame():
    module = import_xense_tactile()
    sensor = FakeSensor()
    calls = []

    def factory(serial_number, *, mac_addr):
        calls.append((serial_number, mac_addr))
        return sensor

    reader = module.XenseTactileReader(
        sensor_serial_number="OG001452",
        mac_addr="1659f0e0dde0",
        sensor_factory=factory,
        output_types=OutputTypes,
        baseline_duration_s=0.0,
    )

    reader.connect()
    frame = reader.read_frame()
    reader.close()

    assert calls == [("OG001452", "1659f0e0dde0")]
    assert reader.baseline_ready is False
    np.testing.assert_array_equal(
        frame.marker_offset,
        np.full(sensor.baseline_marker.shape, 2.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(frame.force_torque, np.arange(6, dtype=np.float64))
    np.testing.assert_array_equal(frame.rectify, sensor.rectify)
    np.testing.assert_array_equal(frame.difference, sensor.difference)
    np.testing.assert_array_equal(frame.depth, sensor.depth)
    assert frame.marker_offset.dtype == np.float32
    assert frame.force_torque.dtype == np.float64
    assert sensor.released is True


def test_reader_uses_median_marker_baseline():
    module = import_xense_tactile()
    sensor = FakeSensor()
    baseline_samples = iter(
        [
            np.full((1, 2, 2), 10.0, dtype=np.float32),
            np.full((1, 2, 2), 2.0, dtype=np.float32),
            np.full((1, 2, 2), 6.0, dtype=np.float32),
        ]
    )
    original_select = sensor.selectSensorInfo

    def select_sensor_info(*outputs):
        if outputs == (OutputTypes.Marker2D,):
            return next(baseline_samples)
        return original_select(*outputs)

    sensor.selectSensorInfo = select_sensor_info
    reader = module.XenseTactileReader(
        sensor_serial_number="OG001452",
        mac_addr="1659f0e0dde0",
        sensor_factory=lambda *_args, **_kwargs: sensor,
        output_types=OutputTypes,
        baseline_duration_s=3.0 / 60.0,
        baseline_rate_hz=60.0,
    )

    reader.connect()
    frame = reader.read_frame()

    np.testing.assert_array_equal(
        frame.marker_offset,
        sensor.frame_marker - 6.0,
    )


def test_reader_rejects_force_resultant_without_six_components():
    module = import_xense_tactile()
    sensor = FakeSensor()
    sensor.force_torque = np.arange(5, dtype=np.float64)
    reader = module.XenseTactileReader(
        sensor_serial_number="OG001452",
        mac_addr="1659f0e0dde0",
        sensor_factory=lambda *_args, **_kwargs: sensor,
        output_types=OutputTypes,
        baseline_duration_s=0.0,
    )

    with pytest.raises(RuntimeError, match="must contain 6 components"):
        reader.read_frame()


def test_reader_releases_sensor_when_baseline_is_invalid():
    module = import_xense_tactile()
    sensor = FakeSensor()
    sensor.baseline_marker = np.zeros((3,), dtype=np.float32)
    reader = module.XenseTactileReader(
        sensor_serial_number="OG001452",
        mac_addr="1659f0e0dde0",
        sensor_factory=lambda *_args, **_kwargs: sensor,
        output_types=OutputTypes,
        baseline_duration_s=0.0,
    )

    with pytest.raises(RuntimeError, match="must end in an x/y dimension"):
        reader.connect()

    assert sensor.released is True
    assert reader.baseline_ready is False
