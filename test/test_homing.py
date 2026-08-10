import importlib
import sys
import types
from pathlib import Path

import numpy as np


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


def test_home_robot_uses_direct_new_rdk_and_preserves_custom_joint_target(monkeypatch):
    calls = []

    class FakeRobot:
        def __init__(self, serial, local_ips):
            calls.append(("Robot", serial, local_ips))
            self._operational = False
            self.q = [0.0] * 7

        def fault(self):
            return False

        def estop_released(self):
            return True

        def operational(self):
            return self._operational

        def Enable(self):
            calls.append(("Enable",))
            self._operational = True

        def SwitchMode(self, mode):
            calls.append(("SwitchMode", mode))

        def info(self):
            return types.SimpleNamespace(q_min=[-3.0] * 7, q_max=[3.0] * 7)

        def SendJointPosition(self, target, velocity, max_velocity, max_acceleration):
            calls.append(
                (
                    "SendJointPosition",
                    target,
                    velocity,
                    max_velocity,
                    max_acceleration,
                )
            )
            self.q = list(target)

        def states(self):
            return types.SimpleNamespace(q=self.q)

        def Stop(self):
            calls.append(("Stop",))

    class FakeTool:
        def __init__(self, robot):
            calls.append(("Tool", robot))

        def exist(self, name):
            calls.append(("tool.exist", name))
            return True

        def Switch(self, name):
            calls.append(("tool.Switch", name))

    fake_rdk = types.SimpleNamespace(
        Robot=FakeRobot,
        Tool=FakeTool,
        Mode=types.SimpleNamespace(
            IDLE="idle",
            NRT_JOINT_IMPEDANCE="joint",
            NRT_PRIMITIVE_EXECUTION="primitive",
        ),
    )
    monkeypatch.setitem(sys.modules, "flexivrdk", fake_rdk)
    sys.modules.pop("homing", None)
    homing = importlib.import_module("homing")

    homing.home_robot(1, wait_interval_s=0.0)

    assert calls[0] == (
        "Robot",
        homing.ROBOT_CONFIGS[1]["serial"],
        homing.ROBOT_CONFIGS[1]["local_ips"],
    )
    assert ("tool.Switch", "tool1") in calls
    move_call = next(call for call in calls if call[0] == "SendJointPosition")
    np.testing.assert_allclose(
        move_call[1],
        np.deg2rad(homing.FIXED_INITIAL_JOINTS_DEG),
    )
    assert calls[-1] == ("Stop",)
