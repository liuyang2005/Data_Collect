"""Direct new-machine Angler and Xense gripper adapters.

The public read/move/close methods match the interfaces used by the existing
collector so the saved robot arrays and collection threads do not change.
"""

import importlib
import math
import threading
import time


def modbus_crc16(message: bytes) -> int:
    crc = 0xFFFF
    for byte in message:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def append_modbus_crc(message: bytes) -> bytes:
    crc = modbus_crc16(message)
    return message + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def _valid_modbus_crc(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    expected = modbus_crc16(frame[:-2])
    actual = frame[-2] | (frame[-1] << 8)
    return actual == expected


class AnglerProtocolError(RuntimeError):
    pass


class AnglerSerial:
    """Read one PDCD Angler encoder directly over Modbus RTU."""

    _FRAME_SIZE = 7

    def __init__(
        self,
        port: str,
        encoder_id: int = 1,
        baudrate: int = 1_000_000,
        inter_request_gap_s: float = 0.002,
        timeout_s: float = 0.03,
        retries: int = 2,
        strict_crc: bool = True,
        serial_factory=None,
    ) -> None:
        if not port:
            raise ValueError("Angler serial port cannot be empty")
        if not 1 <= int(encoder_id) <= 247:
            raise ValueError("Angler encoder ID must be within [1, 247]")
        if baudrate <= 0:
            raise ValueError("Angler baudrate must be positive")
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("Angler timeout must be finite and positive")
        if inter_request_gap_s < 0 or not math.isfinite(inter_request_gap_s):
            raise ValueError("Angler request gap must be finite and non-negative")
        if retries < 0:
            raise ValueError("Angler retries must be non-negative")
        self.port = port
        self.encoder_id = int(encoder_id)
        self.baudrate = int(baudrate)
        self.inter_request_gap_s = float(inter_request_gap_s)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.strict_crc = bool(strict_crc)
        self._serial_factory = serial_factory
        self._serial = None
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            if self._serial is not None:
                raise RuntimeError(f"Angler serial port is already open: {self.port}")
            factory = self._serial_factory
            if factory is None:
                factory = importlib.import_module("serial").Serial
            serial_port = factory(
                port=self.port,
                baudrate=self.baudrate,
                timeout=min(self.timeout_s, 0.01),
                write_timeout=self.timeout_s,
            )
            if hasattr(serial_port, "is_open") and not serial_port.is_open:
                serial_port.open()
            try:
                serial_port.reset_input_buffer()
                serial_port.reset_output_buffer()
            except BaseException:
                serial_port.close()
                raise
            self._serial = serial_port

    @staticmethod
    def _request() -> bytes:
        # Address zero requests a broadcast response from the configured encoder.
        return append_modbus_crc(bytes((0, 3, 0, 65, 0, 1)))

    def read_angle(self) -> float:
        with self._lock:
            if self._serial is None:
                raise RuntimeError("Angler serial port is not open")
            last_error = None
            for _ in range(self.retries + 1):
                try:
                    return self._read_angle_once()
                except Exception as exc:
                    last_error = exc
            raise AnglerProtocolError(
                f"Failed to read Angler on {self.port}: {last_error}"
            ) from last_error

    def _read_angle_once(self) -> float:
        serial_port = self._serial
        serial_port.reset_input_buffer()
        request = self._request()
        written = serial_port.write(request)
        if written is not None and written != len(request):
            raise AnglerProtocolError("Short Angler serial write")
        if hasattr(serial_port, "flush"):
            serial_port.flush()
        if self.inter_request_gap_s:
            time.sleep(self.inter_request_gap_s)

        buffer = bytearray()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            chunk = serial_port.read(self._FRAME_SIZE - len(buffer))
            if chunk:
                buffer.extend(chunk)
            while len(buffer) >= self._FRAME_SIZE:
                frame = bytes(buffer[: self._FRAME_SIZE])
                if frame[0] != self.encoder_id or frame[1:3] != bytes((3, 2)):
                    del buffer[0]
                    continue
                if self.strict_crc and not _valid_modbus_crc(frame):
                    del buffer[0]
                    continue
                raw_angle = (frame[3] << 8) | frame[4]
                return 360.0 * raw_angle / 4096.0
        raise AnglerProtocolError(
            f"Timed out waiting for encoder {self.encoder_id} on {self.port}"
        )

    def close(self) -> None:
        with self._lock:
            serial_port = self._serial
            self._serial = None
            if serial_port is not None:
                serial_port.close()


class AnglerGripperController:
    """Map an Angler angle to the legacy master gripper width interface."""

    def __init__(
        self,
        angler,
        open_angle: float,
        close_angle: float,
        open_width: float,
        close_width: float,
    ) -> None:
        if open_angle == close_angle:
            raise ValueError("open_angle and close_angle must be different")
        self.angler = angler
        self.open_angle = float(open_angle)
        self.close_angle = float(close_angle)
        self.open_width = float(open_width)
        self.close_width = float(close_width)

    def read(self) -> float:
        angle = self.angler.read_angle()
        ratio = (angle - self.close_angle) / (self.open_angle - self.close_angle)
        ratio = min(max(ratio, 0.0), 1.0)
        return self.close_width + ratio * (self.open_width - self.close_width)

    def close(self) -> None:
        self.angler.close()


class XenseGripperAdapter:
    """Expose the direct xensegripper SDK through legacy read/move/close calls."""

    MAX_WIDTH_M = 0.085

    def __init__(
        self,
        mac_addr: str,
        name: str = "Xense",
        velocity_m_s: float = 0.08,
        force_n: float = 20.0,
        connect_attempts: int = 2,
        connect_retry_delay_s: float = 3.0,
        sdk_module=None,
    ) -> None:
        if not mac_addr:
            raise ValueError("Xense gripper MAC/device ID cannot be empty")
        if not 0 < velocity_m_s <= 0.35:
            raise ValueError("Xense velocity must be within (0, 0.35] m/s")
        if not 0 < force_n <= 60:
            raise ValueError("Xense force must be within (0, 60] N")
        if connect_attempts < 1:
            raise ValueError("Xense connect attempts must be positive")
        self.mac_addr = mac_addr
        self.name = name
        self.velocity_m_s = float(velocity_m_s)
        self.force_n = float(force_n)
        self.connect_attempts = int(connect_attempts)
        self.connect_retry_delay_s = float(connect_retry_delay_s)
        self._sdk_module = sdk_module
        self._backend = None
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            if self._backend is not None:
                raise RuntimeError(f"Xense gripper is already open: {self.mac_addr}")
            sdk = self._sdk_module or importlib.import_module("xensegripper")
            last_error = None
            for attempt in range(1, self.connect_attempts + 1):
                try:
                    backend = sdk.XenseGripper.create(mac_addr=self.mac_addr)
                    if backend is None:
                        raise RuntimeError("XenseGripper.create returned None")
                    self._backend = backend
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < self.connect_attempts and self.connect_retry_delay_s > 0:
                        time.sleep(self.connect_retry_delay_s)
            raise RuntimeError(
                f"Failed to connect Xense gripper {self.mac_addr}: {last_error}"
            ) from last_error

    @staticmethod
    def _position_mm(status) -> float:
        try:
            value = status["position"] if isinstance(status, dict) else status.position
            position_mm = float(value)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Xense gripper status: {status!r}") from exc
        if not math.isfinite(position_mm):
            raise RuntimeError("Xense gripper returned a non-finite position")
        return position_mm

    def read(self) -> float:
        with self._lock:
            if self._backend is None:
                raise RuntimeError("Xense gripper is not open")
            width_m = self._position_mm(self._backend.get_gripper_status()) / 1000.0
        if not -0.005 <= width_m <= self.MAX_WIDTH_M + 0.005:
            raise RuntimeError(f"Implausible Xense gripper width: {width_m:.6f} m")
        return width_m

    def move(self, width_m: float):
        target = float(width_m)
        if not math.isfinite(target) or not 0 <= target <= self.MAX_WIDTH_M:
            raise ValueError(f"Xense target width must be within [0, {self.MAX_WIDTH_M}] m")
        with self._lock:
            if self._backend is None:
                raise RuntimeError("Xense gripper is not open")
            return self._backend.set_position(
                target * 1000.0,
                self.velocity_m_s * 1000.0,
                self.force_n,
            )

    def close(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
            if backend is not None:
                backend.release()
