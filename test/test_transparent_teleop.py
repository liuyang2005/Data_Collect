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

    def Init(self):
        self.calls.append(("Init",))

    def Start(self):
        self.calls.append(("Start",))

    def SetWrenchFeedbackScalingFactor(self, pair_idx, factor):
        self.calls.append(("SetWrenchFeedbackScalingFactor", pair_idx, factor))

    def robot_states(self, pair_idx):
        self.calls.append(("robot_states", pair_idx))
        leader = types.SimpleNamespace(q=[1.0] * 7)
        follower = types.SimpleNamespace(q=[2.0] * 7)
        return leader, follower

    def SetLeaderNullSpacePosture(self, pair_idx, posture):
        self.calls.append(("SetLeaderNullSpacePosture", pair_idx, posture))

    def SetFollowerNullSpacePosture(self, pair_idx, posture):
        self.calls.append(("SetFollowerNullSpacePosture", pair_idx, posture))

    def Engage(self, pair_idx, activated):
        self.calls.append(("Engage", pair_idx, activated))


def import_transparent_teleop(monkeypatch):
    fake_module = types.SimpleNamespace(
        TransparentCartesianTeleopLAN=FakeTransparentCartesianTeleopLAN
    )
    monkeypatch.setitem(sys.modules, "flexivtdk", fake_module)
    sys.modules.pop("transparent_teleop", None)
    return importlib.import_module("transparent_teleop")


def test_sets_no_feedback_with_flexivtdk_1_6_signature(monkeypatch):
    module = import_transparent_teleop(monkeypatch)
    pair = module.TransparentCartesianTeleopPair("leader", "follower")
    pair.init()

    pair.set_wrench_feedback_scale(0.0)
    pair.activate(True)

    assert pair.cart_teleop.calls == [
        ("Init",),
        ("Start",),
        ("SetWrenchFeedbackScalingFactor", 0, 0.0),
        ("robot_states", 0),
        ("SetLeaderNullSpacePosture", 0, [1.0] * 7),
        ("SetFollowerNullSpacePosture", 0, [2.0] * 7),
        ("Engage", 0, True),
    ]


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
    assert 'SLAVE_GRIPPER_ID="d254505bfaaa"' in launcher
    assert 'TACTILE_SENSOR_SN="OG000451"' in launcher
