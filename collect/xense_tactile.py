from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


SensorFactory = Callable[..., Any]


@dataclass(frozen=True)
class XenseFingertipFrame:
    timestamp_host_s: float
    marker_offset: np.ndarray
    force_torque: np.ndarray
    force_norm: np.ndarray
    rectify: np.ndarray
    difference: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class XenseTactileFrame:
    left: XenseFingertipFrame
    right: XenseFingertipFrame


class XenseTactileReader:
    """Read complete tactile frames from two Xense fingertip sensors."""

    def __init__(
        self,
        left_sensor_serial_number: str,
        right_sensor_serial_number: str,
        mac_addr: str,
        *,
        sensor_factory: SensorFactory | None = None,
        output_types: Any | None = None,
        baseline_duration_s: float = 1.0,
        baseline_rate_hz: float = 60.0,
    ) -> None:
        left_serial = left_sensor_serial_number.strip()
        right_serial = right_sensor_serial_number.strip()
        if not left_serial:
            raise ValueError("left Xense sensor serial number must be non-empty")
        if not right_serial:
            raise ValueError("right Xense sensor serial number must be non-empty")
        if left_serial == right_serial:
            raise ValueError("left and right Xense sensor serial numbers must be different")
        if not mac_addr.strip():
            raise ValueError("Xense connection identifier must be non-empty")
        if baseline_duration_s < 0.0 or baseline_rate_hz <= 0.0:
            raise ValueError("baseline duration must be non-negative and rate positive")
        self.left_sensor_serial_number = left_serial
        self.right_sensor_serial_number = right_serial
        self.mac_addr = mac_addr.strip()
        self._sensor_factory = sensor_factory
        self._output_types = output_types
        self.baseline_duration_s = float(baseline_duration_s)
        self.baseline_rate_hz = float(baseline_rate_hz)
        self._left_sensor: Any | None = None
        self._right_sensor: Any | None = None
        self._left_marker_reference: np.ndarray | None = None
        self._right_marker_reference: np.ndarray | None = None

    @property
    def baseline_ready(self) -> bool:
        return (
            self._left_marker_reference is not None
            and self._right_marker_reference is not None
        )

    def connect(self) -> None:
        if self._left_sensor is not None or self._right_sensor is not None:
            if (
                self._left_sensor is not None
                and self._right_sensor is not None
                and self.baseline_ready
            ):
                return
            raise RuntimeError("Xense tactile reader is only partially connected")
        if self._sensor_factory is None:
            from xensesdk import Sensor  # type: ignore[import-not-found]

            self._sensor_factory = Sensor.create
            self._output_types = Sensor.OutputType
        if self._output_types is None:
            raise RuntimeError("Xense output types are required")
        try:
            self._left_sensor = self._sensor_factory(
                self.left_sensor_serial_number,
                mac_addr=self.mac_addr,
            )
            self._right_sensor = self._sensor_factory(
                self.right_sensor_serial_number,
                mac_addr=self.mac_addr,
            )
            self._left_marker_reference = self._establish_marker_reference(
                self._left_sensor,
                "left",
            )
            self._right_marker_reference = self._establish_marker_reference(
                self._right_sensor,
                "right",
            )
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def read_frame(self) -> XenseTactileFrame:
        self.connect()
        if (
            self._left_sensor is None
            or self._right_sensor is None
            or self._left_marker_reference is None
            or self._right_marker_reference is None
        ):
            raise RuntimeError("both Xense tactile sensors must be ready")

        left = self._read_fingertip(
            self._left_sensor,
            self._left_marker_reference,
            "left",
        )
        right = self._read_fingertip(
            self._right_sensor,
            self._right_marker_reference,
            "right",
        )
        return XenseTactileFrame(left=left, right=right)

    def close(self) -> None:
        sensors = (self._right_sensor, self._left_sensor)
        self._left_sensor = None
        self._right_sensor = None
        self._left_marker_reference = None
        self._right_marker_reference = None

        first_error: Exception | None = None
        for sensor in sensors:
            if sensor is None:
                continue
            try:
                sensor.release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _read_fingertip(
        self,
        sensor: Any,
        marker_reference: np.ndarray,
        side: str,
    ) -> XenseFingertipFrame:
        outputs = self._output_types
        values = sensor.selectSensorInfo(
            outputs.Marker2D,
            outputs.ForceResultant,
            outputs.ForceNorm,
            outputs.Rectify,
            outputs.Difference,
            outputs.Depth,
        )
        timestamp_host_s = time.time()
        if not isinstance(values, (tuple, list)) or len(values) != 6:
            raise RuntimeError(
                f"{side} Xense complete tactile read must return six outputs"
            )
        marker, force_torque, force_norm, rectify, difference, depth = values

        marker_array = np.asarray(marker, dtype=np.float32)
        if marker_array.shape != marker_reference.shape:
            raise RuntimeError(
                f"{side} Xense marker shape changed from "
                f"{marker_reference.shape} to {marker_array.shape}"
            )
        force_array = np.asarray(force_torque, dtype=np.float64).reshape(-1)
        if force_array.size != 6:
            raise RuntimeError(
                f"{side} Xense ForceResultant must contain 6 components, "
                f"got {force_array.size}"
            )
        if not np.all(np.isfinite(force_array)):
            raise RuntimeError(
                f"{side} Xense ForceResultant contains a non-finite value"
            )

        force_norm_array = np.asarray(force_norm)
        if force_norm_array.ndim != 3 or force_norm_array.shape[-1] != 3:
            raise RuntimeError(
                f"{side} Xense ForceNorm must have shape (H, W, 3), "
                f"got {force_norm_array.shape}"
            )
        if not np.issubdtype(force_norm_array.dtype, np.number):
            raise RuntimeError(f"{side} Xense ForceNorm must be numeric")
        if not np.all(np.isfinite(force_norm_array)):
            raise RuntimeError(
                f"{side} Xense ForceNorm contains a non-finite value"
            )

        return XenseFingertipFrame(
            timestamp_host_s=timestamp_host_s,
            marker_offset=np.asarray(
                marker_array - marker_reference,
                dtype=np.float32,
            ),
            force_torque=force_array,
            force_norm=np.ascontiguousarray(force_norm_array.copy()),
            rectify=_required_image(rectify, f"{side} rectify"),
            difference=_required_image(difference, f"{side} difference"),
            depth=_required_image(depth, f"{side} depth"),
        )

    def _establish_marker_reference(self, sensor: Any, side: str) -> np.ndarray:
        outputs = self._output_types
        sample_count = max(1, round(self.baseline_duration_s * self.baseline_rate_hz))
        period_s = 1.0 / self.baseline_rate_hz
        samples: list[np.ndarray] = []
        next_sample_s = time.monotonic()
        for index in range(sample_count):
            marker = np.asarray(
                sensor.selectSensorInfo(outputs.Marker2D),
                dtype=np.float32,
            )
            if marker.ndim < 2 or marker.shape[-1] != 2:
                raise RuntimeError(
                    f"{side} Xense Marker2D must end in an x/y dimension, "
                    f"got {marker.shape}"
                )
            if samples and marker.shape != samples[0].shape:
                raise RuntimeError(
                    f"{side} Xense marker shape changed during baseline sampling"
                )
            samples.append(marker.copy())
            if index + 1 < sample_count:
                next_sample_s += period_s
                time.sleep(max(0.0, next_sample_s - time.monotonic()))
        return np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)


def _required_image(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Xense {name} image is missing")
    image = np.asarray(value)
    if image.ndim not in (2, 3) or image.size == 0:
        raise RuntimeError(f"Xense {name} image has invalid shape {image.shape}")
    return np.ascontiguousarray(image)


__all__ = ["XenseFingertipFrame", "XenseTactileFrame", "XenseTactileReader"]
