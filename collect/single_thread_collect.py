from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dual_collect_utils import (
    read_robot_sample,
    save_force_stream,
    save_robot_streams,
    save_tactile_stream,
)


@dataclass(frozen=True)
class CycleSample:
    cycle_index: int
    cycle_timestamp_host_s: float
    scheduled_monotonic_ns: int
    acquisition_started_monotonic_ns: int
    acquisition_completed_monotonic_ns: int
    camera_frames: dict[str, tuple[np.ndarray, np.ndarray]]
    tcp_pose: np.ndarray
    tcp_vel: np.ndarray
    q: np.ndarray
    ext_wrench_in_tcp: np.ndarray
    tactile_frame: Any | None
    source_completed_host_s: dict[str, float]


class AlignedRateControl:
    def __init__(self, rate_hz: int) -> None:
        if rate_hz <= 0:
            raise ValueError("collection rate must be positive")
        self.period_ns = round(1_000_000_000 / rate_hz)
        self.next_deadline_ns = time.monotonic_ns()

    def wait_next(self) -> int:
        now_ns = time.monotonic_ns()
        if now_ns < self.next_deadline_ns:
            time.sleep((self.next_deadline_ns - now_ns) / 1_000_000_000)
        scheduled_ns = self.next_deadline_ns
        self.next_deadline_ns += self.period_ns

        now_ns = time.monotonic_ns()
        if self.next_deadline_ns < now_ns:
            missed_periods = (now_ns - self.next_deadline_ns) // self.period_ns + 1
            self.next_deadline_ns += missed_periods * self.period_ns
        return scheduled_ns


def acquire_cycle(
    *,
    cycle_index: int,
    cycle_timestamp_host_s: float,
    scheduled_monotonic_ns: int,
    cameras: Mapping[str, Any],
    state_reader: Any,
    slave_gripper: Any,
    tactile_reader: Any | None,
    use_gripper: bool,
    host_clock: Callable[[], float] = time.time,
    monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
) -> CycleSample:
    acquisition_started_monotonic_ns = monotonic_clock_ns()
    camera_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_completed_host_s: dict[str, float] = {}

    for name, camera in cameras.items():
        color, depth = camera.get()
        if color is None or depth is None:
            raise RuntimeError(f"Camera {name} returned an incomplete RGB-D frame")
        camera_frames[name] = (
            np.asarray(color).copy(),
            np.asarray(depth).copy(),
        )
        source_completed_host_s[f"camera_{name}"] = float(host_clock())

    tcp_xyz, tcp_quat_xyzw, joint_angles, tcp_vel, ext_wrench_in_tcp = (
        read_robot_sample(state_reader)
    )
    source_completed_host_s["robot"] = float(host_clock())

    if use_gripper:
        gripper_width = float(slave_gripper.read())
        source_completed_host_s["gripper"] = float(host_clock())
    else:
        gripper_width = 0.0
    tcp_pose = np.concatenate((tcp_xyz, tcp_quat_xyzw, [gripper_width]))
    q = np.concatenate((joint_angles, [gripper_width]))

    tactile_frame = tactile_reader.read_frame() if tactile_reader is not None else None
    if tactile_frame is not None:
        source_completed_host_s["tactile_left"] = float(
            tactile_frame.left.timestamp_host_s
        )
        source_completed_host_s["tactile_right"] = float(
            tactile_frame.right.timestamp_host_s
        )

    return CycleSample(
        cycle_index=int(cycle_index),
        cycle_timestamp_host_s=float(cycle_timestamp_host_s),
        scheduled_monotonic_ns=int(scheduled_monotonic_ns),
        acquisition_started_monotonic_ns=acquisition_started_monotonic_ns,
        acquisition_completed_monotonic_ns=monotonic_clock_ns(),
        camera_frames=camera_frames,
        tcp_pose=tcp_pose,
        tcp_vel=np.asarray(tcp_vel, dtype=np.float64).copy(),
        q=q,
        ext_wrench_in_tcp=np.asarray(ext_wrench_in_tcp, dtype=np.float64).copy(),
        tactile_frame=tactile_frame,
        source_completed_host_s=source_completed_host_s,
    )


class _EpisodeBuffer:
    def __init__(
        self,
        camera_names: Sequence[str],
        tactile_enabled: bool,
        period_ns: int,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.tactile_enabled = tactile_enabled
        self.period_ns = int(period_ns)
        self.samples: list[CycleSample] = []
        self.write_started_host_s: list[float] = []
        self.write_completed_host_s: list[float] = []

    def commit(
        self,
        sample: CycleSample,
        write_started_host_s: float,
        completion_clock: Callable[[], float],
    ) -> None:
        self.samples.append(sample)
        self.write_started_host_s.append(float(write_started_host_s))
        self.write_completed_host_s.append(float(completion_clock()))

    def save(self, session_dir: str) -> None:
        timestamps = [sample.cycle_timestamp_host_s for sample in self.samples]
        for camera_name in self.camera_names:
            camera_dir = os.path.join(session_dir, camera_name)
            os.makedirs(camera_dir, exist_ok=True)
            np.save(
                os.path.join(camera_dir, "timestamps_host_s.npy"),
                np.asarray(timestamps, dtype=np.float64),
            )

        save_robot_streams(
            session_dir=session_dir,
            tcp_pose_rows=[sample.tcp_pose for sample in self.samples],
            tcp_vel_rows=[sample.tcp_vel for sample in self.samples],
            q_rows=[sample.q for sample in self.samples],
            timestamps_host_s=timestamps,
        )
        save_force_stream(
            session_dir=session_dir,
            ext_wrench_rows=[sample.ext_wrench_in_tcp for sample in self.samples],
            timestamps_host_s=timestamps,
        )

        if self.tactile_enabled:
            for side in ("left", "right"):
                fingertips = [
                    getattr(sample.tactile_frame, side) for sample in self.samples
                ]
                save_tactile_stream(
                    session_dir=session_dir,
                    side=side,
                    marker_offset_rows=[frame.marker_offset for frame in fingertips],
                    force_torque_rows=[frame.force_torque for frame in fingertips],
                    force_norm_rows=[frame.force_norm for frame in fingertips],
                    timestamps_host_s=timestamps,
                )

        timing_dir = os.path.join(session_dir, "timing")
        os.makedirs(timing_dir, exist_ok=True)
        timing_arrays = {
            "cycle_index": [sample.cycle_index for sample in self.samples],
            "cycle_timestamps_host_s": timestamps,
            "scheduled_monotonic_ns": [
                sample.scheduled_monotonic_ns for sample in self.samples
            ],
            "acquisition_started_monotonic_ns": [
                sample.acquisition_started_monotonic_ns for sample in self.samples
            ],
            "acquisition_completed_monotonic_ns": [
                sample.acquisition_completed_monotonic_ns for sample in self.samples
            ],
            "cycle_duration_s": [
                (
                    sample.acquisition_completed_monotonic_ns
                    - sample.acquisition_started_monotonic_ns
                )
                / 1_000_000_000
                for sample in self.samples
            ],
            "deadline_overrun_s": [
                max(
                    0,
                    sample.acquisition_completed_monotonic_ns
                    - sample.scheduled_monotonic_ns
                    - self.period_ns,
                )
                / 1_000_000_000
                for sample in self.samples
            ],
            "write_started_host_s": self.write_started_host_s,
            "write_completed_host_s": self.write_completed_host_s,
        }
        integer_fields = {
            "cycle_index",
            "scheduled_monotonic_ns",
            "acquisition_started_monotonic_ns",
            "acquisition_completed_monotonic_ns",
        }
        for name, values in timing_arrays.items():
            dtype = np.int64 if name in integer_fields else np.float64
            np.save(
                os.path.join(timing_dir, f"{name}.npy"),
                np.asarray(values, dtype=dtype),
            )

        source_names = sorted(
            {
                source_name
                for sample in self.samples
                for source_name in sample.source_completed_host_s
            }
        )
        source_timestamps = {
            source_name: np.asarray(
                [
                    sample.source_completed_host_s.get(source_name, np.nan)
                    for sample in self.samples
                ],
                dtype=np.float64,
            )
            for source_name in source_names
        }
        np.savez(
            os.path.join(timing_dir, "source_completed_host_s.npz"),
            **source_timestamps,
        )


def _image_paths(session_dir: str, sample: CycleSample):
    for camera_name, (color, depth) in sample.camera_frames.items():
        camera_dir = os.path.join(session_dir, camera_name)
        color_path = os.path.join(
            camera_dir, "color", f"{sample.cycle_index:016d}.png"
        )
        depth_path = os.path.join(
            camera_dir, "depth", f"{sample.cycle_index:016d}.png"
        )
        yield color_path, color
        yield depth_path, depth

    if sample.tactile_frame is None:
        return
    filename = f"{sample.cycle_index:06d}.png"
    for side in ("left", "right"):
        fingertip = getattr(sample.tactile_frame, side)
        for image_name in ("rectify", "difference", "depth"):
            yield (
                os.path.join(
                    session_dir,
                    "tactile",
                    side,
                    image_name,
                    filename,
                ),
                getattr(fingertip, image_name),
            )


def _write_cycle_images(session_dir: str, sample: CycleSample, image_writer) -> None:
    written_paths: list[str] = []
    try:
        for path, image in _image_paths(session_dir, sample):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            written_paths.append(path)
            if not image_writer(path, image):
                raise IOError(f"Failed to write aligned cycle image: {path}")
    except BaseException:
        for path in written_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def _run_writer(
    *,
    session_dir: str,
    frame_queue,
    sentinel,
    episode_buffer: _EpisodeBuffer,
    image_writer,
    errors: list[BaseException],
    stop_event,
) -> None:
    writer_failed = False
    try:
        while True:
            item = frame_queue.get()
            try:
                if item is sentinel:
                    return
                if writer_failed:
                    continue
                write_started_host_s = time.time()
                _write_cycle_images(session_dir, item, image_writer)
                episode_buffer.commit(
                    item,
                    write_started_host_s,
                    time.time,
                )
            except BaseException as exc:
                errors.append(exc)
                stop_event.set()
                writer_failed = True
            finally:
                frame_queue.task_done()
    finally:
        try:
            episode_buffer.save(session_dir)
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()


def collect_aligned_teleop_data(
    *,
    state_reader,
    slave_gripper,
    cameras: Mapping[str, Any],
    session_dir: str,
    stop_event,
    fps: int = 10,
    use_gripper: bool = True,
    tactile_reader=None,
    status_period: int = 100,
    writer_queue_size: int = 8,
) -> None:
    import cv2

    if fps <= 0:
        raise ValueError("collection FPS must be positive")
    if writer_queue_size <= 0:
        raise ValueError("writer queue size must be positive")

    rate_control = AlignedRateControl(fps)
    frame_queue = queue.Queue(maxsize=writer_queue_size)
    sentinel = object()
    writer_errors: list[BaseException] = []
    episode_buffer = _EpisodeBuffer(
        camera_names=cameras.keys(),
        tactile_enabled=tactile_reader is not None,
        period_ns=round(1_000_000_000 / fps),
    )
    writer_thread = threading.Thread(
        target=_run_writer,
        kwargs={
            "session_dir": session_dir,
            "frame_queue": frame_queue,
            "sentinel": sentinel,
            "episode_buffer": episode_buffer,
            "image_writer": cv2.imwrite,
            "errors": writer_errors,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    writer_thread.start()

    collection_error: BaseException | None = None
    cycle_index = 0
    try:
        while not stop_event.is_set():
            scheduled_monotonic_ns = rate_control.wait_next()
            if stop_event.is_set():
                break
            cycle_timestamp_host_s = time.time()
            sample = acquire_cycle(
                cycle_index=cycle_index,
                cycle_timestamp_host_s=cycle_timestamp_host_s,
                scheduled_monotonic_ns=scheduled_monotonic_ns,
                cameras=cameras,
                state_reader=state_reader,
                slave_gripper=slave_gripper,
                tactile_reader=tactile_reader,
                use_gripper=use_gripper,
            )
            if writer_errors:
                raise writer_errors[0]
            try:
                frame_queue.put_nowait(sample)
            except queue.Full as exc:
                raise RuntimeError(
                    "Aligned writer queue is full; disk writing cannot sustain the "
                    f"configured {fps} Hz collection rate"
                ) from exc

            if status_period and cycle_index % status_period == 0:
                print(f"Aligned collection rate: {fps} Hz, cycles: {cycle_index}")
            cycle_index += 1
    except BaseException as exc:
        collection_error = exc
        stop_event.set()
    finally:
        frame_queue.put(sentinel)
        writer_thread.join()

    if collection_error is not None:
        raise collection_error
    if writer_errors:
        raise writer_errors[0]
