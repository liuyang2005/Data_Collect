# Home After Recording Design

## Goal

Keep one data-collection process running while returning both robots to
`FIXED_INITIAL_JOINTS_DEG` after every successfully saved recording. Cameras,
the Angler controller, the follower gripper, and both tactile sensors remain
connected across recordings.

## Resource Lifetimes

Application-level resources are initialized once and closed only during final
program cleanup:

- cameras;
- master Angler controller;
- follower Xense gripper;
- left and right Xense tactile sensors.

The transparent TDK teleoperation pair and its `TeleopSlaveStateReader` belong
to one teleoperation cycle. They are stopped and discarded before homing, then
created again after homing. The recording worker and recorder belong to one
episode and must be fully joined and saved before the reset begins.

## Control Flow

When `--home-after-recording` is enabled, `run_keyboard_loop()` returns one of
two outcomes:

- `RESET` after `v` has stopped and successfully saved the current recording;
- `QUIT` after `q` is pressed.

When the option is disabled, `v` preserves the current behavior and continues
inside the same keyboard loop after saving.

`main()` owns an outer teleoperation-cycle loop:

1. Create and configure `TransparentCartesianTeleopPair`.
2. Create the matching `TeleopSlaveStateReader`.
3. Run the existing keyboard loop.
4. On `RESET`, disengage teleoperation and leave the TDK context so `Stop()`
   releases robot control.
5. Home robot IDs 1 and 2 with the existing fixed-joint homing routine.
6. If both robots reach their targets, create the next TDK cycle and wait for
   the operator to press `r` again.
7. On `QUIT`, leave the outer loop and run the existing one-time cleanup.

The program does not automatically engage teleoperation after homing. Requiring
`r` preserves the existing operator confirmation and prevents immediate motion
coupling after reset.

## Configuration

Add an explicit `--home-after-recording` boolean option. The normal collection
launcher enables it. Direct invocations can disable it to retain the current
behavior where `v` saves and immediately returns to the keyboard loop.

The existing homing robot IDs, delay, retry count, and retry delay are reused.
Exit homing remains independently controlled by `--home-on-exit` and must not
cause an extra home cycle after every recording.

## Failure Handling

Automatic homing starts only after `stop_collection()` completes successfully.
TDK robot control must be stopped before any RDK homing object is created or
used. If saving, homing, or rebuilding TDK fails, the program does not proceed
to another recording; it enters the existing final cleanup and exits nonzero.

The follower gripper stays connected and its width is not changed by the
per-recording reset. Existing startup and final-exit gripper behavior remains
unchanged.

## Scope

Do not change recording formats, sampling rates, feedback behavior, keyboard
meanings, tactile output, camera configuration, or fixed homing joint values.
Do not attempt to home through RDK while the transparent TDK controller still
owns the robots.

## Verification

Focused tests use fakes to verify:

- `v` saves before returning `RESET`;
- application-level devices initialize and close exactly once across multiple
  recordings;
- each reset stops TDK before homing both configured robot IDs;
- the next TDK pair is created only after homing succeeds;
- teleoperation is not engaged automatically after reset;
- save, home, and TDK-rebuild failures prevent the next recording and preserve
  cleanup;
- `q` exits without creating another TDK cycle.

Run the focused pytest suite, Python compilation, shell syntax validation, and
`git diff --check`. Real-hardware validation must confirm controller handoff,
homing motion safety, and repeated TDK reinitialization on the Linux collector.
