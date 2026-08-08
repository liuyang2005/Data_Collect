import importlib
import sys
from pathlib import Path

import pytest


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


def import_gripper_devices():
    sys.modules.pop("gripper_devices", None)
    return importlib.import_module("gripper_devices")


class FakeSerialPort:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.writes = []
        self.closed = False

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass

    def read(self, _size):
        return self.chunks.pop(0) if self.chunks else b""

    def close(self):
        self.closed = True


def test_angler_reads_crc_checked_angle_and_closes_port():
    module = import_gripper_devices()
    raw_angle = 1024
    response = module.append_modbus_crc(
        bytes((1, 3, 2, (raw_angle >> 8) & 0xFF, raw_angle & 0xFF))
    )
    port = FakeSerialPort([response])
    angler = module.AnglerSerial(
        port="mock://angler",
        encoder_id=1,
        timeout_s=0.01,
        retries=0,
        serial_factory=lambda **_kwargs: port,
    )

    angler.open()
    angle = angler.read_angle()
    angler.close()

    assert angle == pytest.approx(90.0)
    assert port.writes == [module.append_modbus_crc(bytes((0, 3, 0, 65, 0, 1)))]
    assert port.closed is True


def test_angler_controller_keeps_legacy_read_interface():
    module = import_gripper_devices()

    class FakeAngler:
        def read_angle(self):
            return 30.0

        def close(self):
            self.closed = True

    angler = FakeAngler()
    controller = module.AnglerGripperController(
        angler=angler,
        open_angle=50.0,
        close_angle=10.0,
        open_width=0.08,
        close_width=0.0,
    )

    assert controller.read() == pytest.approx(0.04)
    controller.close()
    assert angler.closed is True


def test_angler_does_not_retry_keyboard_interrupt():
    module = import_gripper_devices()

    class InterruptPort(FakeSerialPort):
        def read(self, _size):
            raise KeyboardInterrupt

    port = InterruptPort([])
    angler = module.AnglerSerial(
        port="mock://angler",
        retries=3,
        serial_factory=lambda **_kwargs: port,
    )
    angler.open()

    with pytest.raises(KeyboardInterrupt):
        angler.read_angle()

    assert len(port.writes) == 1
