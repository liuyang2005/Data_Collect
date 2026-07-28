from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


SensorFactory = Callable[..., Any]


@dataclass(frozen=True)
class XenseTactileFrame:
    marker_offset: np.ndarray
    force_torque: np.ndarray
    rectify: np.ndarray
    difference: np.ndarray
    depth: np.ndarray


class XenseTactileReader:
    """Read complete tactile frames from one Xense fingertip sensor."""

    def __init__(
        self,
        sensor_serial_number: str,
        mac_addr: str,
        *,
        sensor_factory: SensorFactory | None = None,
        output_types: Any | None = None,
        baseline_duration_s: float = 1.0,
        baseline_rate_hz: float = 60.0,
    ) -> None:
        if not sensor_serial_number.strip():
            raise ValueError("Xense sensor serial number must be non-empty")
        if not mac_addr.strip():
            raise ValueError("Xense MAC address must be non-empty")
        if baseline_duration_s < 0.0 or baseline_rate_hz <= 0.0:
            raise ValueError("baseline duration must be non-negative and rate positive")
        self.sensor_serial_number = sensor_serial_number
        self.mac_addr = mac_addr
        self._sensor_factory = sensor_factory
        self._output_types = output_types
        self.baseline_duration_s = float(baseline_duration_s)
        self.baseline_rate_hz = float(baseline_rate_hz)
        self._sensor: Any | None = None
        self._marker_reference: np.ndarray | None = None

    @property
    def baseline_ready(self) -> bool:
        return self._marker_reference is not None

    def connect(self) -> None:
        if self._sensor is not None:
            return
        if self._sensor_factory is None:
            from xensesdk import Sensor  # type: ignore[import-not-found]

            self._sensor_factory = Sensor.create
            self._output_types = Sensor.OutputType
        if self._output_types is None:
            raise RuntimeError("Xense output types are required")
        try:
            self._sensor = self._sensor_factory(
                self.sensor_serial_number,
                mac_addr=self.mac_addr,
            )
            self._establish_marker_reference()
        except Exception:
            self.close()
            raise

    def read_frame(self) -> XenseTactileFrame:
        self.connect()
        if self._sensor is None or self._marker_reference is None:
            raise RuntimeError("Xense tactile sensor is not ready")

        outputs = self._output_types
        values = self._sensor.selectSensorInfo(
            outputs.Marker2D,
            outputs.ForceResultant,
            outputs.Rectify,
            outputs.Difference,
            outputs.Depth,
        )
        if not isinstance(values, (tuple, list)) or len(values) != 5:
            raise RuntimeError("Xense complete tactile read must return five outputs")
        marker, force_torque, rectify, difference, depth = values

        marker_array = np.asarray(marker, dtype=np.float32)
        if marker_array.shape != self._marker_reference.shape:
            raise RuntimeError(
                "Xense marker shape changed from "
                f"{self._marker_reference.shape} to {marker_array.shape}"
            )
        force_array = np.asarray(force_torque, dtype=np.float64).reshape(-1)
        if force_array.size != 6:
            raise RuntimeError(
                "Xense ForceResultant must contain 6 components, "
                f"got {force_array.size}"
            )
        if not np.all(np.isfinite(force_array)):
            raise RuntimeError("Xense force/torque contains a non-finite value")

        return XenseTactileFrame(
            marker_offset=np.asarray(
                marker_array - self._marker_reference,
                dtype=np.float32,
            ),
            force_torque=force_array,
            rectify=_required_image(rectify, "rectify"),
            difference=_required_image(difference, "difference"),
            depth=_required_image(depth, "depth"),
        )

    def close(self) -> None:
        sensor = self._sensor
        self._sensor = None
        self._marker_reference = None
        if sensor is not None:
            sensor.release()

    def _establish_marker_reference(self) -> None:
        assert self._sensor is not None
        outputs = self._output_types
        sample_count = max(1, round(self.baseline_duration_s * self.baseline_rate_hz))
        period_s = 1.0 / self.baseline_rate_hz
        samples: list[np.ndarray] = []
        next_sample_s = time.monotonic()
        for index in range(sample_count):
            marker = np.asarray(
                self._sensor.selectSensorInfo(outputs.Marker2D),
                dtype=np.float32,
            )
            if marker.ndim < 2 or marker.shape[-1] != 2:
                raise RuntimeError(
                    f"Xense Marker2D must end in an x/y dimension, got {marker.shape}"
                )
            if samples and marker.shape != samples[0].shape:
                raise RuntimeError("Xense marker shape changed during baseline sampling")
            samples.append(marker.copy())
            if index + 1 < sample_count:
                next_sample_s += period_s
                time.sleep(max(0.0, next_sample_s - time.monotonic()))
        self._marker_reference = np.median(
            np.stack(samples, axis=0),
            axis=0,
        ).astype(np.float32)


def _required_image(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Xense {name} image is missing")
    image = np.asarray(value)
    if image.ndim not in (2, 3) or image.size == 0:
        raise RuntimeError(f"Xense {name} image has invalid shape {image.shape}")
    return np.ascontiguousarray(image)


__all__ = ["XenseTactileFrame", "XenseTactileReader"]
