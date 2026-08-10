# Dual Xense ForceNorm Collection Design

## Goal

Extend the existing independent-rate Xense tactile stream from one fingertip to
two fingertips while preserving all non-tactile collection behavior. Add the
SDK-provided `ForceNorm` normal-force field to the existing tactile outputs.

## Device Mapping

- Left fingertip sensor: `OG001453`
- Right fingertip sensor: `OG001455`
- Shared Xense gripper identifier: `gripper_8a429d6ea337`
- Target tactile rate: 60 Hz

The shared identifier is passed through the existing `mac_addr` argument used by
`Sensor.create`. Linux hardware validation must confirm that the installed SDK
accepts this identifier for both sensor serial numbers.

## Per-Fingertip Frame

Each fingertip requests these six outputs in one `selectSensorInfo` call:

1. `Marker2D`
2. `ForceResultant`
3. `ForceNorm`
4. `Rectify`
5. `Difference`
6. `Depth`

`ForceNorm` is the raw SDK normal-force component field. It is not derived from
`ForceResultant`. The collector validates that it is a finite numeric 3-D array
whose final dimension is 3. The spatial dimensions are discovered at runtime
instead of being hard-coded because SDK versions can expose different marker or
force-grid sizes.

Each fingertip establishes its own median `Marker2D` baseline. Persisted
`marker_offset` remains the current marker array minus that fingertip's baseline.

## Collection Architecture

One `XenseTactileReader` owns the left and right SDK sensors. The existing
independent tactile worker runs at the configured tactile rate and reads the two
sensors sequentially during each cycle. Each fingertip's six outputs are
same-frame because they come from one SDK call; the two physical sensors are not
claimed to be simultaneous.

The collector records a host timestamp immediately after each fingertip read.
Left and right streams therefore have their own timestamp files even when both
samples belong to the same collection cycle.

Both fingertip image triplets are queued as one dual-fingertip unit. A row is
committed only after all six images (`rectify`, `difference`, and `depth` for
both sides) are written successfully. If any image write fails, partial images
from that unit are removed, both sides discard the corresponding array row, the
shared stop event is set, and the worker exception reaches the main stop path.

## Storage Contract

Each recording session stores:

```text
tactile/
  left/
    marker_offset.npy
    force_torque.npy
    force_norm.npy
    timestamps_host_s.npy
    rectify/*.png
    difference/*.png
    depth/*.png
  right/
    marker_offset.npy
    force_torque.npy
    force_norm.npy
    timestamps_host_s.npy
    rectify/*.png
    difference/*.png
    depth/*.png
```

For each side, all four NPY arrays have the same leading row count, which also
matches the number of committed image triplets. `force_torque.npy` retains the
existing six-component `ForceResultant` values and units. `force_norm.npy`
retains the SDK field values without replacing them with a scalar norm.

The metadata records both serial numbers, the shared gripper identifier, the
effective tactile rate, and all left/right stream paths.

## CLI And Launcher

Replace the single-sensor option with explicit left and right serial-number
options. Keep one shared tactile connection identifier and one tactile-rate
option. The launcher defaults become:

- left serial: `OG001453`
- right serial: `OG001455`
- slave gripper/shared tactile identifier: `gripper_8a429d6ea337`
- tactile rate: 60 Hz

The command-line parser rejects enabled tactile collection when either serial or
the shared identifier is empty.

## Scope Boundaries

Do not change camera, robot state, wrist wrench, feedback, gripper-width,
teleoperation, exit homing, or ACP post-processing behavior. Existing cleanup
continues to release both tactile sensors during normal exit and exceptions.

## Verification

Test-first implementation covers:

- creation and release of both SDK sensors;
- one independent marker baseline per fingertip;
- one six-output SDK call per fingertip frame;
- raw `ForceNorm` validation and persistence;
- symmetric left/right directories and metadata;
- matching arrays and committed image counts;
- cleanup and exception propagation after partial image-write failure;
- CLI validation and launcher defaults.

Run focused pytest suites, Python compilation, shell syntax validation, and
`git diff --check` on Windows. Real Linux hardware validation remains required
for SDK connectivity, observed per-fingertip rate, output shapes, and sustained
dual-sensor recording.
