import json
import os
import queue
import threading
import time
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FPS = 30
MAIN_CAMERA_SERIAL = "104122061018"
WRIST_CAMERA_SERIAL = "260322275475"
# Keep these existing serials until the devices on the new machine are checked.
D415_CAMERAS = {
    f"cam_{MAIN_CAMERA_SERIAL}": MAIN_CAMERA_SERIAL,
    f"cam_{WRIST_CAMERA_SERIAL}_wrist": WRIST_CAMERA_SERIAL,
}
CAMERA_PROFILES = {
    f"cam_{MAIN_CAMERA_SERIAL}": {
        "serial": MAIN_CAMERA_SERIAL,
        "model": "D415",
        "width": 640,
        "height": 480,
    },
    f"cam_{WRIST_CAMERA_SERIAL}_wrist": {
        "serial": WRIST_CAMERA_SERIAL,
        "model": "D405",
        "width": 640,
        "height": 480,
    },
}


def configure_headless_input_backend() -> None:
    """Configure optional input backends on SSH sessions without a display."""
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYNPUT_BACKEND", "dummy")


class RealSenseD415:
    """RealSense D415/D405 wrapper preserving the existing collector interface."""

    def __init__(
        self,
        serial: str,
        fps: int = FPS,
        name: Optional[str] = None,
        model: str = "D415",
        width: int = 640,
        height: int = 480,
    ) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = serial
        self.name = name or serial
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.depth_enabled = True
        self.inpaint = False
        self.hole_filling = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial is not None:
            self.config.enable_device(serial)
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.align = rs.align(rs.stream.color)
        self.pipeline_profile = self.pipeline.start(self.config)

        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            device = self.pipeline_profile.get_device()
            device.hardware_reset()
            time.sleep(5)
            frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame().as_video_frame()
        depth_frame = frames.get_depth_frame().as_depth_frame()
        depth2color = depth_frame.get_profile().get_extrinsics_to(
            color_frame.get_profile()
        )
        self.depth2color = np.eye(4)
        self.depth2color[:3, :3] = np.array(depth2color.rotation).reshape((3, 3))
        self.depth2color[:3, 3] = depth2color.translation

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame().as_video_frame()
        color_intrinsics = (
            color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )
        self.color_intrinsics = [
            color_intrinsics.ppx,
            color_intrinsics.ppy,
            color_intrinsics.fx,
            color_intrinsics.fy,
        ]
        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        self.color_image_dtype = color_image.dtype
        self.color_image_shape = color_image.shape
        self.depth_image_dtype = depth_image.dtype
        self.depth_image_shape = depth_image.shape

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame().as_video_frame()
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        if self.hole_filling is not None:
            depth_frame = self.hole_filling.process(depth_frame)
        color = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        return color, depth

    def close(self) -> None:
        self.pipeline.stop()

    def __del__(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def create_session_dirs(
    save_root: str,
    d415_cameras: Optional[Mapping[str, str]] = None,
    session_name: Optional[str] = None,
) -> str:
    os.makedirs(save_root, exist_ok=True)

    if session_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"record_{timestamp}"

    session_dir = os.path.join(save_root, session_name)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "robot"), exist_ok=True)

    # camera directories
    if d415_cameras:
        for cam_name in d415_cameras.keys():
            cam_dir = os.path.join(session_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            os.makedirs(os.path.join(cam_dir, "color"), exist_ok=True)
            os.makedirs(os.path.join(cam_dir, "depth"), exist_ok=True)

    print(f"Data will be saved to: {session_dir}")
    return session_dir


def write_metadata(session_dir: str, metadata: Any) -> str:
    if isinstance(metadata, Namespace):
        metadata = vars(metadata)
    else:
        metadata = dict(metadata)

    metadata.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    return metadata_path


def init_cameras(
    d415_cameras: Optional[Mapping[str, Any]] = None,
    fps: int = FPS,
) -> Dict[str, RealSenseD415]:
    if d415_cameras is None:
        d415_cameras = CAMERA_PROFILES

    cameras = {}
    try:
        for cam_name, config in d415_cameras.items():
            if isinstance(config, str):
                config = {"serial": config}
            cameras[cam_name] = RealSenseD415(
                serial=config["serial"],
                fps=fps,
                name=cam_name,
                model=config.get("model", "D415"),
                width=config.get("width", 640),
                height=config.get("height", 480),
            )
    except BaseException:
        for camera in reversed(list(cameras.values())):
            try:
                camera.close()
            except Exception:
                pass
        raise
    return cameras


def init_xense(gripper_id: str, name: str = "Xense"):
    from gripper_devices import XenseGripperAdapter

    gripper = XenseGripperAdapter(
        mac_addr=gripper_id,
        name=name,
        force_n=30.0,
    )
    gripper.open()
    return gripper


def init_angler_controller(
    encoder_id: str,
    index: int,
    baudrate: int,
    gap: float,
    strict: bool,
    open_angle: float,
    close_angle: float,
    open_width: float,
    close_width: float,
    name: str = "master_angler",
) -> Any:
    from gripper_devices import AnglerGripperController, AnglerSerial

    angler = AnglerSerial(
        port=encoder_id,
        encoder_id=index,
        baudrate=baudrate,
        inter_request_gap_s=gap if gap >= 0 else 0.002,
        strict_crc=strict,
    )
    angler.open()
    return AnglerGripperController(
        angler=angler,
        open_angle=open_angle,
        close_angle=close_angle,
        open_width=open_width,
        close_width=close_width,
    )


def enqueue_camera_frames(
    cameras: Mapping[str, Any],
    frame_queue,
    frame_idx: int,
) -> list[str]:
    queued_cameras = []
    for name, cam in cameras.items():
        color_frame, depth_frame = cam.get()

        if color_frame is not None and depth_frame is not None:
            frame_queue.put(
                (
                    name,
                    frame_idx,
                    np.asarray(color_frame).copy(),
                    np.asarray(depth_frame).copy(),
                )
            )
            queued_cameras.append(name)
    return queued_cameras


def write_camera_frames(session_dir: str, frame_queue, sentinel, errors, stop_event) -> None:
    import cv2

    while True:
        item = frame_queue.get()
        try:
            if item is sentinel:
                return

            name, frame_idx, color_frame, depth_frame = item
            cv2.imwrite(
                os.path.join(session_dir, name, "color", f"{frame_idx:016d}.png"),
                color_frame,
            )
            cv2.imwrite(
                os.path.join(session_dir, name, "depth", f"{frame_idx:016d}.png"),
                depth_frame,
            )
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()
        finally:
            frame_queue.task_done()


def save_camera_timestamps(
    session_dir: str,
    camera_timestamps: Mapping[str, Sequence[float]],
) -> None:
    for camera_name, timestamps in camera_timestamps.items():
        np.save(
            os.path.join(session_dir, camera_name, "timestamps_host_s.npy"),
            np.asarray(timestamps, dtype=np.float64),
        )


def _array_row_count(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    array = np.load(path, mmap_mode="r")
    return int(array.shape[0]) if array.ndim > 0 else 1


def _png_count(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        entry.is_file() and entry.name.lower().endswith(".png")
        for entry in os.scandir(path)
    )


def summarize_episode(session_dir: str, camera_names: Sequence[str]) -> dict:
    cameras = {}
    for camera_name in camera_names:
        camera_dir = os.path.join(session_dir, camera_name)
        cameras[camera_name] = {
            "color": _png_count(os.path.join(camera_dir, "color")),
            "depth": _png_count(os.path.join(camera_dir, "depth")),
        }

    return {
        "cameras": cameras,
        "robot": {
            "tcp_pose": _array_row_count(
                os.path.join(session_dir, "robot", "tcp_pose.npy")
            ),
            "tcp_vel": _array_row_count(
                os.path.join(session_dir, "robot", "tcp_vel.npy")
            ),
            "q": _array_row_count(os.path.join(session_dir, "robot", "q.npy")),
        },
        "force": _array_row_count(
            os.path.join(session_dir, "ext_wrench_in_tcp.npy")
        ),
        "tactile": _array_row_count(
            os.path.join(session_dir, "tactile", "left", "force_torque.npy")
        ),
    }


def collect_camera_stream(
    cameras: Mapping[str, Any],
    session_dir: str,
    stop_event,
    camera_fps: int,
    status_period: int = 100,
) -> None:
    rate_control = RateControl(camera_fps)
    frame_idx = 0
    camera_timestamps = {name: [] for name in cameras}
    frame_queue = queue.Queue()
    writer_errors = []
    sentinel = object()
    writer_thread = threading.Thread(
        target=write_camera_frames,
        args=(session_dir, frame_queue, sentinel, writer_errors, stop_event),
        daemon=True,
    )
    writer_thread.start()

    try:
        while not stop_event.is_set():
            actual_rate = rate_control.sleep()
            queued_cameras = enqueue_camera_frames(cameras, frame_queue, frame_idx)
            sample_time = time.time()
            for camera_name in queued_cameras:
                camera_timestamps[camera_name].append(sample_time)

            if status_period and frame_idx % status_period == 0:
                print(
                    f"Camera rate: {actual_rate:.2f} Hz, collected frames: {frame_idx}"
                )

            frame_idx += 1
    finally:
        frame_queue.put(sentinel)
        writer_thread.join()

    save_camera_timestamps(session_dir, camera_timestamps)
    if writer_errors:
        raise writer_errors[0]


def write_tactile_images(
    session_dir: str,
    frame_queue,
    sentinel,
    errors,
    stop_event,
    image_writer,
    committed_indices,
) -> None:
    tactile_dir = os.path.join(session_dir, "tactile")
    writer_failed = False
    while True:
        item = frame_queue.get()
        paths = ()
        try:
            if item is sentinel:
                return
            if writer_failed:
                continue

            frame_idx, left, right = item
            filename = f"{frame_idx:06d}.png"
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
                for side, name, _ in images
            ]
            for path, (_, _, image) in zip(paths, images):
                if not image_writer(path, image):
                    raise IOError(f"Failed to write tactile image: {path}")
            committed_indices.append(frame_idx)
        except BaseException as exc:
            errors.append(exc)
            for path in paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_exc:
                    errors.append(cleanup_exc)
            stop_event.set()
            writer_failed = True
        finally:
            frame_queue.task_done()


def save_tactile_stream(
    session_dir: str,
    side: str,
    marker_offset_rows: Sequence[np.ndarray],
    force_torque_rows: Sequence[np.ndarray],
    force_norm_rows: Sequence[np.ndarray],
    timestamps_host_s: Sequence[float],
) -> None:
    if side not in ("left", "right"):
        raise ValueError(f"invalid tactile side: {side}")
    tactile_dir = os.path.join(session_dir, "tactile", side)
    os.makedirs(tactile_dir, exist_ok=True)
    marker_offset = np.asarray(marker_offset_rows, dtype=np.float32)
    force_torque = np.asarray(force_torque_rows, dtype=np.float64)
    force_norm = np.asarray(force_norm_rows)
    timestamps = np.asarray(timestamps_host_s, dtype=np.float64)

    if marker_offset.size == 0:
        marker_offset = np.empty((0, 0, 0, 2), dtype=np.float32)
    if force_torque.size == 0:
        force_torque = np.empty((0, 6), dtype=np.float64)
    if force_norm.size == 0:
        force_norm = np.empty((0, 0, 0, 3), dtype=np.float32)

    np.save(os.path.join(tactile_dir, "marker_offset.npy"), marker_offset)
    np.save(os.path.join(tactile_dir, "force_torque.npy"), force_torque)
    np.save(os.path.join(tactile_dir, "force_norm.npy"), force_norm)
    np.save(os.path.join(tactile_dir, "timestamps_host_s.npy"), timestamps)


def collect_tactile_stream(
    tactile_reader,
    session_dir: str,
    stop_event,
    tactile_fps: int,
    status_period: int = 100,
    image_queue_size: int = 128,
) -> None:
    import cv2

    if image_queue_size <= 0:
        raise ValueError("tactile image queue size must be positive")
    tactile_dir = os.path.join(session_dir, "tactile")
    for side in ("left", "right"):
        for name in ("rectify", "difference", "depth"):
            os.makedirs(os.path.join(tactile_dir, side, name), exist_ok=True)

    rate_control = RateControl(tactile_fps)
    rows = {
        side: {
            "marker_offset": [],
            "force_torque": [],
            "force_norm": [],
            "timestamps_host_s": [],
        }
        for side in ("left", "right")
    }
    writer_errors = []
    committed_indices = []
    frame_queue = queue.Queue(maxsize=image_queue_size)
    sentinel = object()
    writer_thread = threading.Thread(
        target=write_tactile_images,
        args=(
            session_dir,
            frame_queue,
            sentinel,
            writer_errors,
            stop_event,
            cv2.imwrite,
            committed_indices,
        ),
        daemon=True,
    )
    writer_thread.start()
    frame_idx = 0

    try:
        while not stop_event.is_set():
            actual_rate = rate_control.sleep()
            frame = tactile_reader.read_frame()
            for side, fingertip in (("left", frame.left), ("right", frame.right)):
                rows[side]["marker_offset"].append(fingertip.marker_offset.copy())
                rows[side]["force_torque"].append(fingertip.force_torque.copy())
                rows[side]["force_norm"].append(fingertip.force_norm.copy())
                rows[side]["timestamps_host_s"].append(
                    float(fingertip.timestamp_host_s)
                )
            try:
                frame_queue.put_nowait((frame_idx, frame.left, frame.right))
            except queue.Full as exc:
                raise RuntimeError("Tactile image writer queue is full") from exc

            if status_period and frame_idx % status_period == 0:
                print(
                    f"Tactile rate: {actual_rate:.2f} Hz, "
                    f"collected frames: {frame_idx}"
                )
            frame_idx += 1
    finally:
        frame_queue.put(sentinel)
        writer_thread.join()
        for side in ("left", "right"):
            side_rows = rows[side]
            save_tactile_stream(
                session_dir=session_dir,
                side=side,
                marker_offset_rows=[
                    side_rows["marker_offset"][i] for i in committed_indices
                ],
                force_torque_rows=[
                    side_rows["force_torque"][i] for i in committed_indices
                ],
                force_norm_rows=[
                    side_rows["force_norm"][i] for i in committed_indices
                ],
                timestamps_host_s=[
                    side_rows["timestamps_host_s"][i] for i in committed_indices
                ],
            )

    if writer_errors:
        raise writer_errors[0]


def tdk_pose_to_saved_xyzquat(tdk_pose: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Flexiv TDK/RDK pose order [x,y,z,qw,qx,qy,qz] to saved xyzw."""
    tdk_pose = np.asarray(tdk_pose, dtype=np.float64).reshape(-1)
    if tdk_pose.shape[0] != 7:
        raise ValueError(f"tcp_pose must have shape (7,), got {tdk_pose.shape}")

    xyz = tdk_pose[:3].copy()
    quat_xyzw = np.array(
        [tdk_pose[4], tdk_pose[5], tdk_pose[6], tdk_pose[3]],
        dtype=np.float64,
    )
    return xyz, quat_xyzw


def _as_1d_array(name: str, value: Any, expected_size: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_size is not None and array.shape[0] != expected_size:
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    return array


def read_robot_sample(
    state_reader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tcp xyz, tcp quat xyzw, joint q, tcp_vel, and ext wrench."""
    if hasattr(state_reader, "read_robot_sample"):
        tcp_xyz, tcp_quat_xyzw, joint_angles, tcp_vel, ext_wrench_in_tcp = (
            state_reader.read_robot_sample()
        )
        return (
            _as_1d_array("tcp_xyz", tcp_xyz, 3),
            _as_1d_array("tcp_quat_xyzw", tcp_quat_xyzw, 4),
            _as_1d_array("joint_angles", joint_angles),
            _as_1d_array("tcp_vel", tcp_vel, 6),
            _as_1d_array("ext_wrench_in_tcp", ext_wrench_in_tcp, 6),
        )

    if hasattr(state_reader, "read_slave_state"):
        slave_state = state_reader.read_slave_state()
        tcp_xyz, tcp_quat_xyzw = tdk_pose_to_saved_xyzquat(slave_state.tcp_pose)
        joint_angles = _as_1d_array("joint_angles", slave_state.q)
        tcp_vel = _as_1d_array("tcp_vel", slave_state.tcp_vel, 6)
        ext_wrench_in_tcp = _as_1d_array(
            "ext_wrench_in_tcp",
            slave_state.ext_wrench_in_tcp,
            6,
        )
        return tcp_xyz, tcp_quat_xyzw, joint_angles, tcp_vel, ext_wrench_in_tcp

    raise AttributeError(
        "state_reader must provide read_robot_sample() or read_slave_state() "
        "with tcp_pose, q, tcp_vel, and ext_wrench_in_tcp fields"
    )


def _read_robot_sample_locked(state_reader, state_reader_lock=None):
    if state_reader_lock is None:
        return read_robot_sample(state_reader)
    with state_reader_lock:
        return read_robot_sample(state_reader)


def save_robot_streams(
    session_dir: str,
    tcp_pose_rows: Sequence[np.ndarray],
    tcp_vel_rows: Sequence[np.ndarray],
    q_rows: Sequence[np.ndarray],
    timestamps_host_s: Sequence[float],
) -> None:
    timestamps = np.asarray(timestamps_host_s, dtype=np.float64)
    tcp_pose = np.asarray(tcp_pose_rows, dtype=np.float64)
    tcp_vel = np.asarray(tcp_vel_rows, dtype=np.float64)
    q = np.asarray(q_rows, dtype=np.float64)

    if tcp_pose.size == 0:
        tcp_pose = np.empty((0, 8), dtype=np.float64)
    if tcp_vel.size == 0:
        tcp_vel = np.empty((0, 6), dtype=np.float64)
    if q.size == 0:
        q = np.empty((0, 8), dtype=np.float64)

    robot_dir = os.path.join(session_dir, "robot")
    os.makedirs(robot_dir, exist_ok=True)
    np.save(os.path.join(robot_dir, "tcp_pose.npy"), tcp_pose)
    np.save(os.path.join(robot_dir, "tcp_vel.npy"), tcp_vel)
    np.save(os.path.join(robot_dir, "q.npy"), q)
    np.save(os.path.join(robot_dir, "timestamps_host_s.npy"), timestamps)


def save_force_stream(
    session_dir: str,
    ext_wrench_rows: Sequence[np.ndarray],
    timestamps_host_s: Sequence[float],
) -> None:
    timestamps = np.asarray(timestamps_host_s, dtype=np.float64)
    ext_wrench = np.asarray(ext_wrench_rows, dtype=np.float64)

    if ext_wrench.size == 0:
        ext_wrench = np.empty((0, 6), dtype=np.float64)

    np.save(os.path.join(session_dir, "ext_wrench_in_tcp.npy"), ext_wrench)
    np.save(
        os.path.join(session_dir, "ext_wrench_in_tcp_timestamps_host_s.npy"),
        timestamps,
    )


def collect_robot_stream(
    state_reader,
    slave_gripper,
    session_dir: str,
    stop_event,
    robot_fps: int,
    use_gripper: bool = True,
    status_period: int = 100,
    state_reader_lock: Optional[threading.Lock] = None,
) -> None:
    rate_control = RateControl(robot_fps)
    frame_idx = 0
    tcp_pose_rows = []
    tcp_vel_rows = []
    q_rows = []
    timestamps_host_s = []

    while not stop_event.is_set():
        actual_rate = rate_control.sleep()

        tcp_xyz, tcp_quat_xyzw, slave_joint_angles, tcp_vel, _ = _read_robot_sample_locked(
            state_reader,
            state_reader_lock,
        )
        sample_time = time.time()
        slave_gripper_width = slave_gripper.read() if use_gripper else 0.0

        pose_data = np.concatenate([tcp_xyz, tcp_quat_xyzw, [slave_gripper_width]])
        joint_data = np.concatenate([slave_joint_angles, [slave_gripper_width]])
        tcp_pose_rows.append(pose_data)
        tcp_vel_rows.append(tcp_vel)
        q_rows.append(joint_data)
        timestamps_host_s.append(sample_time)

        if status_period and frame_idx % status_period == 0:
            print(f"Robot rate: {actual_rate:.2f} Hz, collected frames: {frame_idx}")

        frame_idx += 1

    save_robot_streams(
        session_dir=session_dir,
        tcp_pose_rows=tcp_pose_rows,
        tcp_vel_rows=tcp_vel_rows,
        q_rows=q_rows,
        timestamps_host_s=timestamps_host_s,
    )


def collect_force_stream(
    state_reader,
    session_dir: str,
    stop_event,
    force_fps: int,
    status_period: int = 100,
    state_reader_lock: Optional[threading.Lock] = None,
) -> None:
    rate_control = RateControl(force_fps)
    frame_idx = 0
    ext_wrench_rows = []
    timestamps_host_s = []

    while not stop_event.is_set():
        actual_rate = rate_control.sleep()

        _, _, _, _, ext_wrench_in_tcp = _read_robot_sample_locked(
            state_reader,
            state_reader_lock,
        )
        sample_time = time.time()
        ext_wrench_rows.append(ext_wrench_in_tcp)
        timestamps_host_s.append(sample_time)

        if status_period and frame_idx % status_period == 0:
            print(f"Force rate: {actual_rate:.2f} Hz, collected frames: {frame_idx}")

        frame_idx += 1

    save_force_stream(
        session_dir=session_dir,
        ext_wrench_rows=ext_wrench_rows,
        timestamps_host_s=timestamps_host_s,
    )


def collect_teleop_data(
    state_reader,
    slave_gripper,
    cameras: Mapping[str, Any],
    session_dir: str,
    stop_event,
    fps: int = FPS,
    use_gripper: bool = True,
    status_period: int = 100,
    camera_fps: Optional[int] = None,
    robot_fps: Optional[int] = None,
    force_fps: Optional[int] = None,
    tactile_reader=None,
    tactile_fps: Optional[int] = None,
) -> None:
    effective_camera_fps = camera_fps if camera_fps is not None else fps
    effective_robot_fps = robot_fps if robot_fps is not None else fps
    effective_force_fps = force_fps if force_fps is not None else effective_robot_fps
    effective_tactile_fps = tactile_fps if tactile_fps is not None else fps
    state_reader_lock = threading.Lock()
    errors = []

    def run_worker(worker, *args, **kwargs):
        try:
            worker(*args, **kwargs)
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()

    print("Start data collection...")

    threads = []
    if cameras:
        threads.append(
            threading.Thread(
                target=run_worker,
                args=(
                    collect_camera_stream,
                    cameras,
                    session_dir,
                    stop_event,
                    effective_camera_fps,
                    status_period,
                ),
                daemon=True,
            )
        )

    threads.append(
        threading.Thread(
            target=run_worker,
            args=(
                collect_robot_stream,
                state_reader,
                slave_gripper,
                session_dir,
                stop_event,
                effective_robot_fps,
                use_gripper,
                status_period,
                state_reader_lock,
            ),
            daemon=True,
        )
    )
    threads.append(
        threading.Thread(
            target=run_worker,
            args=(
                collect_force_stream,
                state_reader,
                session_dir,
                stop_event,
                effective_force_fps,
                status_period,
                state_reader_lock,
            ),
            daemon=True,
        )
    )
    if tactile_reader is not None:
        threads.append(
            threading.Thread(
                target=run_worker,
                args=(
                    collect_tactile_stream,
                    tactile_reader,
                    session_dir,
                    stop_event,
                    effective_tactile_fps,
                    status_period,
                ),
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]


class RateControl:
    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        self.last_time = time.time()
        self.actual_rate = 0
        self.frame_count = 0
        self.start_time = time.time()

    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.interval - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

        self.last_time = time.time()
        self.frame_count += 1

        total_time = time.time() - self.start_time
        if total_time > 0:
            self.actual_rate = self.frame_count / total_time

        return self.actual_rate
