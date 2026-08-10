import importlib
import sys
import types
from pathlib import Path


COLLECT_DIR = Path(__file__).resolve().parents[1] / "collect"
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))


def import_dual_collect(monkeypatch):
    monkeypatch.setitem(sys.modules, "flexivrdk", types.SimpleNamespace())
    sys.modules.pop("dual_collect", None)
    return importlib.import_module("dual_collect")


class FakeCollectThread:
    def is_alive(self):
        return True


class FakeTeleopPair:
    def __init__(self):
        self.activate_calls = []
        self.sync_calls = 0

    def any_fault(self):
        return False

    def is_stopped(self):
        return False

    def activate(self, activated):
        self.activate_calls.append(activated)

    def sync_null_space_postures(self):
        self.sync_calls += 1


def run_keyboard_with_keys(monkeypatch, home_after_recording, keys):
    dual_collect = import_dual_collect(monkeypatch)
    events = []
    keys = iter(keys)
    termios_module = types.SimpleNamespace(
        TCSADRAIN="drain",
        tcgetattr=lambda _stream: "old-settings",
        tcsetattr=lambda *_args: events.append("restore_terminal"),
    )
    tty_module = types.SimpleNamespace(setcbreak=lambda _fd: None)
    monkeypatch.setitem(sys.modules, "termios", termios_module)
    monkeypatch.setitem(sys.modules, "tty", tty_module)
    monkeypatch.setattr(dual_collect, "_read_key_nonblocking", lambda: next(keys))
    monkeypatch.setattr(dual_collect.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        dual_collect,
        "start_recording",
        lambda *_args, **_kwargs: (
            events.append("start_recording") or "session",
            object(),
            FakeCollectThread(),
        ),
    )
    monkeypatch.setattr(
        dual_collect,
        "stop_collection",
        lambda *_args, **_kwargs: events.append("stop_collection"),
    )
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)

    outcome = dual_collect.run_keyboard_loop(
        args=types.SimpleNamespace(home_after_recording=home_after_recording),
        teleop_pair=FakeTeleopPair(),
        state_reader=object(),
        cameras={},
        master_gripper=None,
        slave_gripper=None,
        tactile_reader=None,
        d415_cameras={},
        tdk_tcp_pose_order="tdk",
        saved_tcp_pose_order="saved",
        gripper_eps=1e-4,
        gripper_wait_time=0.0,
        null_space_period=0.0,
        use_gripper=False,
    )
    return dual_collect, events, outcome


def test_v_returns_reset_only_after_recording_is_saved(monkeypatch):
    dual_collect, events, outcome = run_keyboard_with_keys(
        monkeypatch,
        home_after_recording=True,
        keys=["c", "v", "q"],
    )

    assert outcome == dual_collect.KEYBOARD_RESET
    assert events == ["start_recording", "stop_collection", "restore_terminal"]


def test_v_continues_same_keyboard_loop_when_auto_home_is_disabled(monkeypatch):
    dual_collect, events, outcome = run_keyboard_with_keys(
        monkeypatch,
        home_after_recording=False,
        keys=["c", "v", "q"],
    )

    assert outcome == dual_collect.KEYBOARD_QUIT
    assert events == ["start_recording", "stop_collection", "restore_terminal"]


def make_homing_args(**overrides):
    values = {
        "home_delay": 0.0,
        "home_robot_ids": "1,2",
        "home_retries": 3,
        "home_retry_delay": 0.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_home_configured_robots_homes_both_ids_in_order(monkeypatch):
    dual_collect = import_dual_collect(monkeypatch)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "homing",
        types.SimpleNamespace(home_robot=lambda robot_id: calls.append(robot_id)),
    )

    result = dual_collect.home_configured_robots(
        make_homing_args(),
        "after recording",
    )

    assert result is True
    assert calls == [1, 2]


def test_home_configured_robots_retries_and_reports_final_failure(monkeypatch):
    dual_collect = import_dual_collect(monkeypatch)
    calls = []

    def fail_second_robot(robot_id):
        calls.append(robot_id)
        if robot_id == 2:
            raise RuntimeError("home failed")

    monkeypatch.setitem(
        sys.modules,
        "homing",
        types.SimpleNamespace(home_robot=fail_second_robot),
    )

    result = dual_collect.home_configured_robots(
        make_homing_args(home_retries=2),
        "after recording",
    )

    assert result is False
    assert calls == [1, 2, 2]


class CountingDevice:
    def __init__(self):
        self.close_count = 0
        self.moves = []

    def move(self, width):
        self.moves.append(width)

    def close(self):
        self.close_count += 1


def make_main_args():
    return types.SimpleNamespace(
        first_sn="master",
        second_sn="slave",
        network_interface=["192.168.10.2"],
        wrench_feedback_scale=0.0,
        use_gripper=True,
        slave_gripper_id="8a429d6ea337",
        angler_id="angler",
        angler_index=1,
        angler_baudrate=1000000,
        angler_gap=0.002,
        angler_strict=True,
        angler_open_angle=349.102,
        angler_close_angle=314.561,
        slave_open_width=0.075,
        slave_close_width=0.001,
        initial_gripper_width=0.075,
        use_tactile=True,
        tactile_left_sensor_sn="OG001453",
        tactile_right_sensor_sn="OG001455",
        tactile_mac_addr="8a429d6ea337",
        camera_fps=30,
        fps=30,
        gripper_eps=1e-4,
        gripper_wait_time=0.0,
        null_space_period=0.0,
        home_after_recording=True,
        home_on_exit=False,
        home_robot_ids="1,2",
        home_delay=0.0,
        home_retries=3,
        home_retry_delay=0.0,
    )


def install_main_fakes(
    monkeypatch,
    keyboard_outcomes,
    home_result=True,
    fail_tdk_enter_cycle=None,
):
    dual_collect = import_dual_collect(monkeypatch)
    dual_collect_utils = importlib.import_module("dual_collect_utils")
    events = []
    outcomes = iter(keyboard_outcomes)
    master_gripper = CountingDevice()
    slave_gripper = CountingDevice()
    camera = CountingDevice()

    class FakeTactileReader:
        instances = []

        def __init__(self, **_kwargs):
            self.connect_count = 0
            self.close_count = 0
            self.instances.append(self)
            events.append("tactile_init")

        def connect(self):
            self.connect_count += 1
            events.append("tactile_connect")

        def close(self):
            self.close_count += 1
            events.append("tactile_close")

    class FakeTeleopPair:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.cycle = len(self.instances) + 1
            self.instances.append(self)
            events.append(f"tdk_construct_{self.cycle}")

        def __enter__(self):
            events.append(f"tdk_enter_{self.cycle}")
            if self.cycle == fail_tdk_enter_cycle:
                raise RuntimeError("TDK rebuild failed")
            return self

        def __exit__(self, *_args):
            events.append(f"tdk_exit_{self.cycle}")

        def set_wrench_feedback_scale(self, _factor):
            events.append(f"tdk_configure_{self.cycle}")

    tactile_module = types.ModuleType("xense_tactile")
    tactile_module.XenseTactileReader = FakeTactileReader
    monkeypatch.setitem(sys.modules, "xense_tactile", tactile_module)

    transparent_module = types.ModuleType("transparent_teleop")
    transparent_module.SAVED_TCP_POSE_ORDER = "saved"
    transparent_module.TDK_TCP_POSE_ORDER = "tdk"
    transparent_module.TransparentCartesianTeleopPair = FakeTeleopPair
    transparent_module.TeleopSlaveStateReader = lambda pair: ("state", pair.cycle)
    monkeypatch.setitem(sys.modules, "transparent_teleop", transparent_module)

    monkeypatch.setattr(dual_collect, "parse_args", make_main_args)
    monkeypatch.setattr(
        dual_collect_utils,
        "init_xense",
        lambda *_args, **_kwargs: events.append("gripper_init") or slave_gripper,
    )
    monkeypatch.setattr(
        dual_collect_utils,
        "init_angler_controller",
        lambda *_args, **_kwargs: events.append("angler_init") or master_gripper,
    )
    monkeypatch.setattr(
        dual_collect_utils,
        "init_cameras",
        lambda *_args, **_kwargs: events.append("camera_init")
        or {"cam_test": camera},
    )
    monkeypatch.setattr(
        dual_collect,
        "run_keyboard_loop",
        lambda _args, pair, *_rest: events.append(f"keyboard_{pair.cycle}")
        or next(outcomes),
    )
    monkeypatch.setattr(
        dual_collect,
        "home_configured_robots",
        lambda _args, _reason: events.append("home") or home_result,
    )
    return types.SimpleNamespace(
        dual_collect=dual_collect,
        events=events,
        teleop_class=FakeTeleopPair,
        tactile_class=FakeTactileReader,
        master_gripper=master_gripper,
        slave_gripper=slave_gripper,
        camera=camera,
    )


def test_main_keeps_devices_alive_across_tdk_reset_cycles(monkeypatch):
    setup = install_main_fakes(
        monkeypatch,
        keyboard_outcomes=["reset", "quit"],
    )

    try:
        setup.dual_collect.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("main() should exit through SystemExit")

    assert len(setup.teleop_class.instances) == 2
    assert setup.events.index("tdk_exit_1") < setup.events.index("home")
    assert setup.events.index("home") < setup.events.index("tdk_construct_2")
    assert setup.events.count("camera_init") == 1
    assert len(setup.tactile_class.instances) == 1
    assert setup.tactile_class.instances[0].connect_count == 1
    assert setup.tactile_class.instances[0].close_count == 1
    assert setup.camera.close_count == 1
    assert setup.master_gripper.close_count == 1
    assert setup.slave_gripper.close_count == 1


def test_main_does_not_start_next_tdk_cycle_when_homing_fails(monkeypatch):
    setup = install_main_fakes(
        monkeypatch,
        keyboard_outcomes=["reset"],
        home_result=False,
    )

    try:
        setup.dual_collect.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("main() should exit through SystemExit")

    assert len(setup.teleop_class.instances) == 1
    assert setup.events.count("home") == 1
    assert setup.camera.close_count == 1


def test_main_cleans_up_when_tdk_rebuild_fails(monkeypatch):
    setup = install_main_fakes(
        monkeypatch,
        keyboard_outcomes=["reset"],
        fail_tdk_enter_cycle=2,
    )

    try:
        setup.dual_collect.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("main() should exit through SystemExit")

    assert len(setup.teleop_class.instances) == 2
    assert setup.events.index("home") < setup.events.index("tdk_enter_2")
    assert setup.camera.close_count == 1
    assert setup.master_gripper.close_count == 1
    assert setup.slave_gripper.close_count == 1
