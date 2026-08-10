import importlib
import sys
import types
from pathlib import Path


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


class FakeTransparentCartesianTeleopLAN:
    def __init__(self, robot_pairs, lan_ips):
        self.robot_pairs = robot_pairs
        self.lan_ips = lan_ips
        self.calls = []

    def Init(self, enabled, zero_mode):
        self.calls.append(("Init", enabled, zero_mode))

    def Start(self):
        self.calls.append(("Start",))

    def SetWrenchFeedbackScalingFactor(self, pair_idx, factor):
        self.calls.append(("SetWrenchFeedbackScalingFactor", pair_idx, factor))

    def instances(self, pair_idx):
        self.calls.append(("instances", pair_idx))
        return self.leader, self.follower

    def SetLeaderNullSpacePosture(self, pair_idx, posture):
        self.calls.append(("SetLeaderNullSpacePosture", pair_idx, posture))

    def SetFollowerNullSpacePosture(self, pair_idx, posture):
        self.calls.append(("SetFollowerNullSpacePosture", pair_idx, posture))

    def Engage(self, pair_idx, activated):
        self.calls.append(("Engage", pair_idx, activated))


def import_transparent_teleop(monkeypatch):
    fake_flexivrdk = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "flexivrdk", fake_flexivrdk)
    leader_state = types.SimpleNamespace(q=[1.0] * 7)
    follower_state = types.SimpleNamespace(
        q=[2.0] * 7,
        tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        tcp_vel=[0.0] * 6,
        ext_wrench_in_tcp=[3.0] * 6,
    )
    leader = types.SimpleNamespace(states=lambda: leader_state)
    follower = types.SimpleNamespace(states=lambda: follower_state)
    fake_module = types.SimpleNamespace(
        TransparentCartesianTeleopLAN=FakeTransparentCartesianTeleopLAN,
        ZeroFTSensor=types.SimpleNamespace(Enable="zero-enabled", Disable="zero-disabled"),
    )
    monkeypatch.setitem(sys.modules, "flexivtdk", fake_module)
    sys.modules.pop("transparent_teleop", None)
    module = importlib.import_module("transparent_teleop")
    module._test_instances = (leader, follower)
    return module


def test_sets_no_feedback_with_flexivtdk_1_6_signature(monkeypatch):
    module = import_transparent_teleop(monkeypatch)
    pair = module.TransparentCartesianTeleopPair("leader", "follower")
    pair.cart_teleop.leader, pair.cart_teleop.follower = module._test_instances
    pair.init()

    pair.set_wrench_feedback_scale(0.0)
    pair.activate(True)

    assert pair.cart_teleop.calls == [
        ("Init", True, "zero-enabled"),
        ("Start",),
        ("instances", 0),
        ("SetWrenchFeedbackScalingFactor", 0, 0.0),
        ("SetLeaderNullSpacePosture", 0, [1.0] * 7),
        ("SetFollowerNullSpacePosture", 0, [2.0] * 7),
        ("Engage", 0, True),
    ]

    assert pair.read_slave_state().ext_wrench_in_tcp == [3.0] * 6


def test_rejects_feedback_scale_before_teleop_start(monkeypatch):
    module = import_transparent_teleop(monkeypatch)
    pair = module.TransparentCartesianTeleopPair("leader", "follower")

    try:
        pair.set_wrench_feedback_scale(0.0)
    except RuntimeError as exc:
        assert "has not been started" in str(exc)
    else:
        raise AssertionError("expected feedback scaling before start to fail")


def test_launcher_selects_no_feedback_and_new_xense_ids():
    launcher = (COLLECT_DIR / "run_dual_collect.sh").read_text(encoding="utf-8")

    assert 'WRENCH_FEEDBACK_SCALE="0.0"' in launcher
    assert '--wrench-feedback-scale "$WRENCH_FEEDBACK_SCALE"' in launcher
    assert 'SLAVE_GRIPPER_ID="8a429d6ea337"' in launcher
    assert 'TACTILE_LEFT_SENSOR_SN="OG001453"' in launcher
    assert 'TACTILE_RIGHT_SENSOR_SN="OG001455"' in launcher
    assert '--tactile-left-sensor-sn "$TACTILE_LEFT_SENSOR_SN"' in launcher
    assert '--tactile-right-sensor-sn "$TACTILE_RIGHT_SENSOR_SN"' in launcher
    assert 'NETWORK_INTERFACES="192.168.97.10"' in launcher
    assert (
        'ANGLER_ID="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"'
        in launcher
    )
    assert 'ANGLER_OPEN_ANGLE="349.102"' in launcher
    assert 'ANGLER_CLOSE_ANGLE="314.561"' in launcher
    assert 'SLAVE_OPEN_WIDTH="0.075"' in launcher
    assert 'SLAVE_CLOSE_WIDTH="0.001"' in launcher
    assert 'HOME_AFTER_RECORDING="true"' in launcher
    assert '--home-after-recording "$HOME_AFTER_RECORDING"' in launcher
