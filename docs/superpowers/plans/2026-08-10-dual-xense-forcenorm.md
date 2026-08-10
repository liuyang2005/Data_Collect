# Dual Xense ForceNorm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect the existing tactile outputs plus raw SDK `ForceNorm` from left sensor `OG001453` and right sensor `OG001455`, storing symmetric per-fingertip streams under `tactile/left` and `tactile/right`.

**Architecture:** One `XenseTactileReader` owns both SDK sensors and establishes an independent marker baseline for each. A single independent-rate tactile worker reads a dual frame, queues both image triplets atomically, and persists left/right NPY rows only for image sets that fully commit.

**Tech Stack:** Python 3.9/3.10, NumPy, Xense SDK `Sensor.OutputType`, OpenCV PNG writing, pytest, POSIX shell launcher.

---

## File Map

- `collect/xense_tactile.py`: dual-sensor lifecycle, per-side baselines, six-output reads, validation, and frame types.
- `collect/dual_collect_utils.py`: dual-side image writing, array persistence, summary path, and independent tactile worker.
- `collect/dual_collect.py`: CLI, metadata, reader construction, and cleanup wiring.
- `collect/run_dual_collect.sh`: machine-specific left/right serials and shared gripper identifier.
- `collect/README.md` and `README.md`: launch examples and symmetric storage contract.
- `test/test_xense_tactile.py`: reader unit tests with fake SDK sensors.
- `test/test_dual_collect_utils.py`: CLI, metadata, runtime wiring, persistence, and failure tests.
- `test/test_transparent_teleop.py`: launcher configuration assertions.

### Task 1: Dual-Sensor Reader And Raw ForceNorm

**Files:**
- Modify: `test/test_xense_tactile.py`
- Modify: `collect/xense_tactile.py`

- [ ] **Step 1: Write failing reader tests**

Extend the fake output enum and sensor so a full read includes raw `ForceNorm`:

```python
class OutputTypes:
    Marker2D = "marker"
    ForceResultant = "force"
    ForceNorm = "force_norm"
    Rectify = "rectify"
    Difference = "difference"
    Depth = "depth"


class FakeSensor:
    def __init__(self, baseline_value, frame_value):
        self.released = False
        self.baseline_marker = np.full((1, 2, 2), baseline_value, dtype=np.float32)
        self.frame_marker = np.full((1, 2, 2), frame_value, dtype=np.float32)
        self.force_torque = np.arange(6, dtype=np.float64) + frame_value
        self.force_norm = np.full((2, 3, 3), frame_value, dtype=np.float32)
        self.rectify = np.full((2, 3, 3), frame_value, dtype=np.uint8)
        self.difference = np.full((2, 3), frame_value, dtype=np.uint8)
        self.depth = np.full((2, 3), frame_value, dtype=np.uint16)

    def selectSensorInfo(self, *outputs):
        if outputs == (OutputTypes.Marker2D,):
            return self.baseline_marker.copy()
        assert outputs == (
            OutputTypes.Marker2D,
            OutputTypes.ForceResultant,
            OutputTypes.ForceNorm,
            OutputTypes.Rectify,
            OutputTypes.Difference,
            OutputTypes.Depth,
        )
        return (
            self.frame_marker.copy(),
            self.force_torque.copy(),
            self.force_norm.copy(),
            self.rectify.copy(),
            self.difference.copy(),
            self.depth.copy(),
        )

    def release(self):
        self.released = True
```

Replace the single-sensor happy-path test with a dual-sensor test that constructs:

```python
reader = module.XenseTactileReader(
    left_sensor_serial_number="OG001453",
    right_sensor_serial_number="OG001455",
    mac_addr="gripper_8a429d6ea337",
    sensor_factory=factory,
    output_types=OutputTypes,
    baseline_duration_s=0.0,
)
```

Assert factory calls are left then right, `frame.left` and `frame.right` contain their own marker offsets, six-axis forces, raw `force_norm`, images, and finite host timestamps, and `close()` releases both sensors. Add focused tests that reject duplicate serials, reject a `ForceNorm` array whose final dimension is not 3, establish different median baselines for left and right, and release the already-created left sensor if right creation or baseline setup fails.

- [ ] **Step 2: Run reader tests and verify the new API fails**

Run:

```powershell
pytest test/test_xense_tactile.py -q
```

Expected: failures show the old constructor does not accept left/right serial arguments and `OutputTypes.ForceNorm` is not requested.

- [ ] **Step 3: Implement dual frame types and reader lifecycle**

Introduce per-side and dual frame dataclasses:

```python
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
```

Change `XenseTactileReader.__init__` to accept distinct non-empty
`left_sensor_serial_number`, `right_sensor_serial_number`, and `mac_addr`.
Store `_left_sensor`, `_right_sensor`, `_left_marker_reference`, and
`_right_marker_reference`. `connect()` creates left then right using:

```python
self._sensor_factory(serial_number, mac_addr=self.mac_addr)
```

Establish a median baseline independently for each sensor. Implement a private
`_read_fingertip(sensor, marker_reference, side)` that makes exactly one call:

```python
values = sensor.selectSensorInfo(
    outputs.Marker2D,
    outputs.ForceResultant,
    outputs.ForceNorm,
    outputs.Rectify,
    outputs.Difference,
    outputs.Depth,
)
```

Require six returned values. Validate marker shape against that side's baseline,
validate a finite six-component `ForceResultant`, and validate `ForceNorm` with:

```python
force_norm_array = np.asarray(force_norm)
if force_norm_array.ndim != 3 or force_norm_array.shape[-1] != 3:
    raise RuntimeError(
        f"{side} Xense ForceNorm must have shape (H, W, 3), "
        f"got {force_norm_array.shape}"
    )
if not np.issubdtype(force_norm_array.dtype, np.number):
    raise RuntimeError(f"{side} Xense ForceNorm must be numeric")
if not np.all(np.isfinite(force_norm_array)):
    raise RuntimeError(f"{side} Xense ForceNorm contains a non-finite value")
```

Copy the raw field without deriving it from `ForceResultant`, record
`time.time()` immediately after the SDK read, and return left then right from
`read_frame()`. `close()` releases right then left, resets both references, and
re-raises the first release error after attempting both releases.

- [ ] **Step 4: Run reader tests and verify green**

Run:

```powershell
pytest test/test_xense_tactile.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the reader change**

```powershell
git add collect/xense_tactile.py test/test_xense_tactile.py
git commit -m "feat: read dual Xense ForceNorm frames"
```

### Task 2: Symmetric Atomic Tactile Persistence

**Files:**
- Modify: `test/test_dual_collect_utils.py`
- Modify: `collect/dual_collect_utils.py`

- [ ] **Step 1: Write failing dual-storage tests**

Update tactile test fixtures to create `XenseTactileFrame(left=..., right=...)`
with `XenseFingertipFrame` values. In the aligned-write test, collect two dual
frames and assert both sides contain:

```python
for side in ("left", "right"):
    side_dir = tmp_path / "tactile" / side
    assert np.load(side_dir / "marker_offset.npy").shape == (2, 1, 2, 2)
    assert np.load(side_dir / "force_torque.npy").shape == (2, 6)
    assert np.load(side_dir / "force_norm.npy").shape == (2, 2, 3, 3)
    assert np.load(side_dir / "timestamps_host_s.npy").shape == (2,)
    for index in range(2):
        filename = f"{index:06d}.png"
        assert (side_dir / "rectify" / filename).exists()
        assert (side_dir / "difference" / filename).exists()
        assert (side_dir / "depth" / filename).exists()
```

Change the incomplete-write test so `cv2.imwrite` fails for the right-side
`difference` image. Assert no PNG remains anywhere and both sides save empty
arrays with shapes `(0, 0, 0, 2)`, `(0, 6)`, `(0, 0, 0, 3)`, and `(0,)`.
Update the multi-rate and reader-failure assertions to expect
`tactile/left/force_torque.npy` and `tactile/right/force_torque.npy`.

- [ ] **Step 2: Run storage tests and verify old paths fail**

Run:

```powershell
pytest test/test_dual_collect_utils.py -q -k "tactile_stream or independent_stream_rates"
```

Expected: failures show the writer still consumes a single flat frame and does not create left/right directories or `force_norm.npy`.

- [ ] **Step 3: Implement dual-side writer and save helpers**

Make `write_tactile_images` unpack one item as `(frame_idx, left, right)` and
build all six paths before writing:

```python
images = []
for side, frame in (("left", left), ("right", right)):
    images.extend(
        (
            (side, "rectify", frame.rectify),
            (side, "difference", frame.difference),
            (side, "depth", frame.depth),
        )
    )
paths = [
    os.path.join(tactile_dir, side, name, filename)
    for side, name, _image in images
]
```

Append the frame index only after all six writes return true. On failure, remove
every path in the dual unit, set the shared stop event, retain the exception,
and drain later queue entries exactly as the current writer does.

Change `save_tactile_stream` to accept a `side` argument and a
`force_norm_rows` sequence. Save below `tactile/<side>`, using explicit empty
arrays when no dual unit committed:

```python
marker_offset = np.empty((0, 0, 0, 2), dtype=np.float32)
force_torque = np.empty((0, 6), dtype=np.float64)
force_norm = np.empty((0, 0, 0, 3), dtype=np.float32)
timestamps = np.empty((0,), dtype=np.float64)
```

In `collect_tactile_stream`, create image directories for both sides, maintain
separate row dictionaries, enqueue both fingertip frames together, filter both
sides with the shared committed-index list, and call `save_tactile_stream` once
per side. Use each `XenseFingertipFrame.timestamp_host_s` rather than generating
one timestamp after both reads. Update session summary counting to read the
atomic left-side path `tactile/left/force_torque.npy`.

- [ ] **Step 4: Run storage tests and verify green**

Run:

```powershell
pytest test/test_dual_collect_utils.py -q -k "tactile_stream or independent_stream_rates"
```

Expected: selected tests pass and partial dual image units leave no persisted rows.

- [ ] **Step 5: Commit persistence changes**

```powershell
git add collect/dual_collect_utils.py test/test_dual_collect_utils.py
git commit -m "feat: store symmetric dual tactile streams"
```

### Task 3: CLI, Metadata, Runtime Wiring, And Cleanup

**Files:**
- Modify: `test/test_dual_collect_utils.py`
- Modify: `collect/dual_collect.py`

- [ ] **Step 1: Write failing CLI and runtime tests**

Replace the single-sensor CLI test with explicit options:

```python
"--tactile-left-sensor-sn", "OG001453",
"--tactile-right-sensor-sn", "OG001455",
"--tactile-mac-addr", "gripper_8a429d6ea337",
```

Assert defaults are `OG001453` and `OG001455`. Parameterize enabled-tactile
parser failures for an empty left serial, empty right serial, duplicate serials,
non-positive tactile FPS, and an empty shared identifier.

Assert metadata contains:

```python
assert metadata["tactile_sensor_serials"] == {
    "left": "OG001453",
    "right": "OG001455",
}
assert metadata["tactile_stream_files"] == {
    "left": {
        "marker_offset": "tactile/left/marker_offset.npy",
        "force_torque": "tactile/left/force_torque.npy",
        "force_norm": "tactile/left/force_norm.npy",
        "timestamps": "tactile/left/timestamps_host_s.npy",
        "rectify": "tactile/left/rectify/*.png",
        "difference": "tactile/left/difference/*.png",
        "depth": "tactile/left/depth/*.png",
    },
    "right": {
        "marker_offset": "tactile/right/marker_offset.npy",
        "force_torque": "tactile/right/force_torque.npy",
        "force_norm": "tactile/right/force_norm.npy",
        "timestamps": "tactile/right/timestamps_host_s.npy",
        "rectify": "tactile/right/rectify/*.png",
        "difference": "tactile/right/difference/*.png",
        "depth": "tactile/right/depth/*.png",
    },
}
```

Update the fake reader in the main lifecycle test to require both serials and
the shared identifier. Assert it connects once, is passed to recording, and is
closed on exit.

- [ ] **Step 2: Run CLI/runtime tests and verify red**

Run:

```powershell
pytest test/test_dual_collect_utils.py -q -k "tactile_settings or tactile_stream_files or connects_passes_and_closes_tactile_reader or build_metadata_describes_tactile"
```

Expected: failures show the old `--tactile-sensor-sn` interface and flat metadata.

- [ ] **Step 3: Implement CLI, metadata, and reader construction**

Replace `--tactile-sensor-sn` with:

```python
parser.add_argument(
    "--tactile-left-sensor-sn",
    default="OG001453",
    help="Left Xense fingertip sensor serial number",
)
parser.add_argument(
    "--tactile-right-sensor-sn",
    default="OG001455",
    help="Right Xense fingertip sensor serial number",
)
```

When tactile collection is enabled, reject blank serials, equal serials, a
blank shared identifier, and a non-positive rate. Add `tactile_sensor_serials`
and nested `tactile_stream_files` to metadata. Construct the reader with:

```python
tactile_reader = XenseTactileReader(
    left_sensor_serial_number=args.tactile_left_sensor_sn,
    right_sensor_serial_number=args.tactile_right_sensor_sn,
    mac_addr=args.tactile_mac_addr,
)
```

Keep the existing `finally` cleanup call; the reader now releases both SDK
sensors internally.

- [ ] **Step 4: Run CLI/runtime tests and verify green**

Run:

```powershell
pytest test/test_dual_collect_utils.py -q -k "tactile_settings or tactile_stream_files or connects_passes_and_closes_tactile_reader or build_metadata_describes_tactile"
```

Expected: selected tests pass.

- [ ] **Step 5: Commit runtime wiring**

```powershell
git add collect/dual_collect.py test/test_dual_collect_utils.py
git commit -m "feat: configure dual Xense tactile sensors"
```

### Task 4: Launcher, Documentation, And Final Verification

**Files:**
- Modify: `test/test_transparent_teleop.py`
- Modify: `collect/run_dual_collect.sh`
- Modify: `collect/README.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing launcher assertions**

Update the launcher test to assert these exact assignments and arguments:

```python
assert 'SLAVE_GRIPPER_ID="gripper_8a429d6ea337"' in launcher
assert 'TACTILE_LEFT_SENSOR_SN="OG001453"' in launcher
assert 'TACTILE_RIGHT_SENSOR_SN="OG001455"' in launcher
assert '--tactile-left-sensor-sn "$TACTILE_LEFT_SENSOR_SN"' in launcher
assert '--tactile-right-sensor-sn "$TACTILE_RIGHT_SENSOR_SN"' in launcher
```

- [ ] **Step 2: Run launcher test and verify red**

Run:

```powershell
pytest test/test_transparent_teleop.py -q
```

Expected: assertions fail against the old single-sensor variables.

- [ ] **Step 3: Update launcher and documentation**

Set:

```sh
SLAVE_GRIPPER_ID="gripper_8a429d6ea337"
TACTILE_LEFT_SENSOR_SN="OG001453"
TACTILE_RIGHT_SENSOR_SN="OG001455"
TACTILE_MAC_ADDR="$SLAVE_GRIPPER_ID"
```

Pass both serial options when tactile is enabled. Update both README files to
describe two complete tactile frames, raw `ForceNorm` semantics, separate host
timestamps, the symmetric directory tree, and the fact that left/right SDK
calls are sequential rather than physically simultaneous. Do not alter robot,
camera, wrench, gripper-width, feedback, homing, or ACP documentation beyond
references necessary to keep the launch example accurate.

- [ ] **Step 4: Run focused and repository verification**

Run:

```powershell
pytest test/test_xense_tactile.py test/test_dual_collect_utils.py test/test_transparent_teleop.py -q
python -m py_compile collect/xense_tactile.py collect/dual_collect_utils.py collect/dual_collect.py
wsl bash -n collect/run_dual_collect.sh
git diff --check
```

Expected: pytest reports zero failures, compilation and shell syntax exit 0,
and `git diff --check` prints no errors. If the full pytest suite imports missing
Windows-only hardware dependencies, report that boundary separately rather than
claiming Linux hardware readiness.

- [ ] **Step 5: Audit scope and commit**

Run:

```powershell
git status --short
git diff --stat HEAD
git diff -- collect/xense_tactile.py collect/dual_collect_utils.py collect/dual_collect.py collect/run_dual_collect.sh collect/README.md README.md test/test_xense_tactile.py test/test_dual_collect_utils.py test/test_transparent_teleop.py
```

Confirm only the planned tactile, launcher, documentation, and test paths changed.
Then commit:

```powershell
git add collect/run_dual_collect.sh collect/README.md README.md test/test_transparent_teleop.py
git commit -m "docs: document dual Xense tactile collection"
```

- [ ] **Step 6: Record hardware validation boundary**

Report that Windows tests prove parsing, storage, cleanup, and fake-SDK behavior.
Do not claim successful `Sensor.create`, observed 60 Hz, runtime shapes, or long
dual-fingertip stability until validated on the Linux collection machine.
