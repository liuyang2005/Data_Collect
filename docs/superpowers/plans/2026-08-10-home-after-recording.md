# Home After Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep collection peripherals connected while automatically homing both robots after each successfully saved trajectory.

**Architecture:** Treat cameras, grippers, and tactile readers as process-level resources, while each transparent TDK pair and state reader belongs to one teleoperation cycle. `run_keyboard_loop()` returns a reset or quit outcome; the outer loop releases TDK before using the existing RDK homing routine and creates the next TDK pair only after homing succeeds.

**Tech Stack:** Python 3, Flexiv RDK/TDK adapters, pytest fakes, POSIX shell launcher.

---

### Task 1: Add Explicit Configuration

**Files:**
- Modify: `collect/dual_collect.py`
- Modify: `collect/run_dual_collect.sh`
- Modify: `test/test_transparent_teleop.py`
- Modify: `test/test_dual_collect_utils.py`

- [ ] **Step 1: Write failing parser and launcher tests**

Add assertions that direct CLI use defaults `home_after_recording` to `False`,
and that the normal launcher defines `HOME_AFTER_RECORDING="true"` and passes
`--home-after-recording "$HOME_AFTER_RECORDING"`.

```python
assert args.home_after_recording is False
assert 'HOME_AFTER_RECORDING="true"' in launcher
assert '--home-after-recording "$HOME_AFTER_RECORDING"' in launcher
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest test/test_dual_collect_utils.py::test_dual_collect_keeps_gripper_enabled_by_default test/test_transparent_teleop.py::test_launcher_selects_no_feedback_and_new_xense_ids -q`

Expected: FAIL because the option and launcher variable do not exist.

- [ ] **Step 3: Add the option and launcher argument**

Add this parser option beside `--home-on-exit`:

```python
parser.add_argument(
    "--home-after-recording",
    type=parse_bool,
    default=False,
    help="Home configured robots after each successfully saved recording",
)
```

Set `HOME_AFTER_RECORDING="true"` in `run_dual_collect.sh` and pass it in the
existing homing argument group.

- [ ] **Step 4: Run the focused tests and verify pass**

Run the same pytest command. Expected: `2 passed`.

### Task 2: Return a Reset Outcome After Saving

**Files:**
- Create: `test/test_dual_collect_lifecycle.py`
- Modify: `collect/dual_collect.py`

- [ ] **Step 1: Write a failing keyboard-loop test**

Use fake `termios`, `tty`, teleop, recording thread, and key sequence `c`, `v`.
Patch `start_recording()` and `stop_collection()` to append to an event list.

```python
outcome = dual_collect.run_keyboard_loop(
    args=args,
    teleop_pair=teleop_pair,
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
assert outcome == dual_collect.KEYBOARD_RESET
assert events == ["start_recording", "stop_collection"]
```

Add a second case with `home_after_recording=False` and keys `c`, `v`, `q` to
verify the same TDK keyboard loop continues until it returns `KEYBOARD_QUIT`.

- [ ] **Step 2: Run the lifecycle tests and verify failure**

Run: `python -m pytest test/test_dual_collect_lifecycle.py -q`

Expected: FAIL because outcome constants and return behavior are missing.

- [ ] **Step 3: Implement outcome constants and return behavior**

Define:

```python
KEYBOARD_RESET = "reset"
KEYBOARD_QUIT = "quit"
```

After `stop_collection()` succeeds in the `v` branch, return
`KEYBOARD_RESET` only when `args.home_after_recording` is true. Return
`KEYBOARD_QUIT` for `q` and when the teleop loop terminates. Keep the existing
`finally` block so recording completion, terminal restoration, and disengage
still occur before control returns.

- [ ] **Step 4: Run the lifecycle tests and verify pass**

Run: `python -m pytest test/test_dual_collect_lifecycle.py -q`

Expected: keyboard-loop cases PASS.

### Task 3: Reuse Homing Retry Logic Between Episodes

**Files:**
- Modify: `collect/dual_collect.py`
- Modify: `test/test_dual_collect_lifecycle.py`

- [ ] **Step 1: Write failing homing-order and failure tests**

Patch `homing.home_robot`, `time.sleep`, and the logger. Verify configured IDs
are called in order and retries are retained. Verify a final failure returns
`False`.

```python
assert dual_collect.home_configured_robots(args, "after recording") is True
assert calls == [1, 2]
```

- [ ] **Step 2: Run the new tests and verify failure**

Run: `python -m pytest test/test_dual_collect_lifecycle.py -q`

Expected: FAIL because `home_configured_robots()` is not defined.

- [ ] **Step 3: Extract a generic homing helper**

Move delay, ID parsing, retries, and logging from `home_robots_on_exit()` into:

```python
def home_configured_robots(args, reason):
    if args.home_delay > 0:
        time.sleep(args.home_delay)

    from homing import home_robot

    ok = True
    for robot_id in parse_home_robot_ids(args.home_robot_ids):
        robot_ok = False
        for attempt in range(1, args.home_retries + 1):
            try:
                logger.info(
                    "Homing robot %d %s (attempt %d/%d)",
                    robot_id,
                    reason,
                    attempt,
                    args.home_retries,
                )
                home_robot(robot_id)
                robot_ok = True
                break
            except Exception as exc:
                if attempt >= args.home_retries:
                    logger.exception(
                        "Failed to home robot %d %s: %s",
                        robot_id,
                        reason,
                        exc,
                    )
                else:
                    logger.warning(
                        "Failed to home robot %d %s on attempt %d/%d: %s; "
                        "retrying in %.1fs",
                        robot_id,
                        reason,
                        attempt,
                        args.home_retries,
                        exc,
                        args.home_retry_delay,
                    )
                    if args.home_retry_delay > 0:
                        time.sleep(args.home_retry_delay)
        ok = ok and robot_ok
    return ok
```

Keep `home_robots_on_exit()` as the independent `args.home_on_exit` gate that
delegates with reason `"on exit"`. The per-recording flow calls the generic
helper only after the TDK context has exited.

- [ ] **Step 4: Run homing and lifecycle tests**

Run: `python -m pytest test/test_homing.py test/test_dual_collect_lifecycle.py -q`

Expected: all tests PASS.

### Task 4: Keep Peripherals Alive Across TDK Reset Cycles

**Files:**
- Modify: `collect/dual_collect.py`
- Modify: `test/test_dual_collect_lifecycle.py`
- Modify: `test/test_dual_collect_utils.py`

- [ ] **Step 1: Write failing process-lifetime tests**

Use fake modules and event logs to make keyboard outcomes return `RESET`, then
`QUIT`. Assert this exact relationship:

```python
assert events.index("tdk_exit_1") < events.index("home")
assert events.index("home") < events.index("tdk_enter_2")
assert camera.init_count == camera.close_count == 1
assert tactile.connect_count == tactile.close_count == 1
assert master_gripper.close_count == slave_gripper.close_count == 1
```

Add failure cases proving a failed home does not construct TDK cycle 2 and a
failed TDK rebuild exits nonzero while closing every process-level device once.

- [ ] **Step 2: Run the lifecycle tests and verify failure**

Run: `python -m pytest test/test_dual_collect_lifecycle.py test/test_dual_collect_utils.py::test_main_connects_passes_and_closes_tactile_reader -q`

Expected: FAIL because peripherals currently initialize inside the one TDK
context and there is no outer cycle.

- [ ] **Step 3: Implement the outer TDK cycle**

Initialize grippers, tactile, and cameras once before the loop. In the loop,
create/configure a fresh `TransparentCartesianTeleopPair`, create its state
reader, and call `run_keyboard_loop()`. After the context exits:

```python
if outcome != KEYBOARD_RESET:
    break
if not home_configured_robots(args, "after recording"):
    raise RuntimeError("Failed to home robots after recording")
```

Leave existing final device cleanup and independently gated exit homing in the
outer `finally` block. Do not move or reset the gripper between episodes.

- [ ] **Step 4: Run lifecycle and existing collector tests**

Run: `python -m pytest test/test_dual_collect_lifecycle.py test/test_dual_collect_utils.py test/test_homing.py test/test_transparent_teleop.py -q`

Expected: all tests PASS.

### Task 5: Document and Verify the Runtime Workflow

**Files:**
- Modify: `README.md`
- Modify: `collect/README.md`

- [ ] **Step 1: Update operator documentation**

Document `HOME_AFTER_RECORDING="true"`, the new `v` sequence, persistent
camera/gripper/tactile connections, TDK release before homing, and the need to
press `r` again after reset. Retain the warning to verify a collision-free
joint-space return path.

- [ ] **Step 2: Run static and focused verification**

Run:

```bash
python -m py_compile collect/dual_collect.py collect/homing.py collect/transparent_teleop.py
python -m pytest test/test_dual_collect_lifecycle.py test/test_dual_collect_utils.py test/test_homing.py test/test_transparent_teleop.py -q
sh -n collect/run_dual_collect.sh
git diff --check
```

Expected: compilation and shell checks exit 0, tests pass, and diff check is
clean. Real robot motion remains explicitly unverified on Windows.

- [ ] **Step 3: Commit the implementation**

```bash
git add collect/dual_collect.py collect/run_dual_collect.sh README.md collect/README.md test/test_dual_collect_lifecycle.py test/test_dual_collect_utils.py test/test_transparent_teleop.py
git commit -m "feat: home robots after each recording"
```
