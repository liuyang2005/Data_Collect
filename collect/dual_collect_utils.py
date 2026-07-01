import json
import os
import time
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FPS = 30
D415_CAMERAS = {
    "cam_327322062498": "327322062498",
}


class RealSenseD415:
    """D415 wrapper with the core behavior used by r3kit's D415 implementation."""

    def __init__(self, serial: str, fps: int = FPS, name: Optional[str] = None) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = serial
        self.name = name or serial
        self.depth_enabled = True
        self.inpaint = False
        self.hole_filling = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial is not None:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
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
    d415_cameras: Optional[Mapping[str, str]] = None,
    fps: int = FPS,
) -> Dict[str, RealSenseD415]:
    if d415_cameras is None:
        d415_cameras = D415_CAMERAS

    return {
        cam_name: RealSenseD415(serial=serial, fps=fps, name=cam_name)
        for cam_name, serial in d415_cameras.items()
    }


def init_xense(gripper_id: str, name: str = "Xense"):
    from xensegripper import XenseGripper

    return XenseGripper.create(mac_addr=gripper_id)


class AnglerGripperController:
    def __init__(
        self,
        encoder,
        open_angle: float,
        close_angle: float,
        open_width: float,
        close_width: float,
    ) -> None:
        if open_angle == close_angle:
            raise ValueError("open_angle and close_angle must be different")
        self.encoder = encoder
        self.open_angle = float(open_angle)
        self.close_angle = float(close_angle)
        self.open_width = float(open_width)
        self.close_width = float(close_width)

    def read(self) -> float:
        angle = float(np.asarray(self.encoder.get()["angle"]).reshape(-1)[0])
        ratio = (angle - self.close_angle) / (self.open_angle - self.close_angle)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return self.close_width + ratio * (self.open_width - self.close_width)


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
) -> AnglerGripperController:
    from r3kit.devices.encoder.pdcd.angler import Angler

    encoder = Angler(
        id=encoder_id,
        index=[index],
        baudrate=baudrate,
        gap=gap,
        strict=strict,
        name=name,
    )
    return AnglerGripperController(
        encoder=encoder,
        open_angle=open_angle,
        close_angle=close_angle,
        open_width=open_width,
        close_width=close_width,
    )


def save_camera_frames(
    cameras: Mapping[str, Any],
    session_dir: str,
    frame_idx: int,
) -> None:
    import cv2

    for name, cam in cameras.items():
        color_frame, depth_frame = cam.get()

        if color_frame is not None and depth_frame is not None:
            cv2.imwrite(
                os.path.join(session_dir, name, "color", f"{frame_idx:016d}.png"),
                color_frame,
            )
            cv2.imwrite(
                os.path.join(session_dir, name, "depth", f"{frame_idx:016d}.png"),
                depth_frame,
            )


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


def read_robot_sample(state_reader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tcp xyz, tcp quat xyzw, joint q, and ext_wrench_in_tcp."""
    if hasattr(state_reader, "read_robot_sample"):
        tcp_xyz, tcp_quat_xyzw, joint_angles, ext_wrench_in_tcp = (
            state_reader.read_robot_sample()
        )
        return (
            _as_1d_array("tcp_xyz", tcp_xyz, 3),
            _as_1d_array("tcp_quat_xyzw", tcp_quat_xyzw, 4),
            _as_1d_array("joint_angles", joint_angles),
            _as_1d_array("ext_wrench_in_tcp", ext_wrench_in_tcp, 6),
        )

    if hasattr(state_reader, "read_slave_state"):
        slave_state = state_reader.read_slave_state()
        tcp_xyz, tcp_quat_xyzw = tdk_pose_to_saved_xyzquat(slave_state.tcp_pose)
        joint_angles = _as_1d_array("joint_angles", slave_state.q)
        ext_wrench_in_tcp = _as_1d_array(
            "ext_wrench_in_tcp",
            slave_state.ext_wrench_in_tcp,
            6,
        )
        return tcp_xyz, tcp_quat_xyzw, joint_angles, ext_wrench_in_tcp

    raise AttributeError(
        "state_reader must provide read_robot_sample() or read_slave_state() "
        "with tcp_pose, q, and ext_wrench_in_tcp fields"
    )


def save_robot_streams(
    session_dir: str,
    tcp_rows: Sequence[np.ndarray],
    joint_rows: Sequence[np.ndarray],
    ext_wrench_rows: Sequence[np.ndarray],
    timestamps_host_s: Sequence[float],
) -> None:
    timestamps = np.asarray(timestamps_host_s, dtype=np.float64)
    tcps = np.asarray(tcp_rows, dtype=np.float64)
    angles = np.asarray(joint_rows, dtype=np.float64)
    ext_wrench = np.asarray(ext_wrench_rows, dtype=np.float64)

    if tcps.size == 0:
        tcps = np.empty((0, 8), dtype=np.float64)
    if angles.size == 0:
        angles = np.empty((0, 8), dtype=np.float64)
    if ext_wrench.size == 0:
        ext_wrench = np.empty((0, 6), dtype=np.float64)

    np.save(os.path.join(session_dir, "tcps.npy"), tcps)
    np.save(os.path.join(session_dir, "angles.npy"), angles)
    np.save(os.path.join(session_dir, "ext_wrench_in_tcp.npy"), ext_wrench)

    np.save(os.path.join(session_dir, "tcps_timestamps_host_s.npy"), timestamps)
    np.save(os.path.join(session_dir, "angles_timestamps_host_s.npy"), timestamps)
    np.save(
        os.path.join(session_dir, "ext_wrench_in_tcp_timestamps_host_s.npy"),
        timestamps,
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
) -> None:
    rate_control = RateControl(fps)
    frame_idx = 0
    tcp_rows = []
    joint_rows = []
    ext_wrench_rows = []
    timestamps_host_s = []

    print("Start data collection...")

    while not stop_event.is_set():
        actual_rate = rate_control.sleep()
        sample_time = time.time()

        # camera data collection
        save_camera_frames(cameras, session_dir, frame_idx)

        # TCP pose, joint angle, ext wrench, and gripper data collection
        tcp_xyz, tcp_quat_xyzw, slave_joint_angles, ext_wrench_in_tcp = (
            read_robot_sample(state_reader)
        )
        slave_gripper_width = slave_gripper.read() if use_gripper else 0.0

        pose_data = np.concatenate([tcp_xyz, tcp_quat_xyzw, [slave_gripper_width]])
        joint_data = np.concatenate([slave_joint_angles, [slave_gripper_width]])
        tcp_rows.append(pose_data)
        joint_rows.append(joint_data)
        ext_wrench_rows.append(ext_wrench_in_tcp)
        timestamps_host_s.append(sample_time)

        if status_period and frame_idx % status_period == 0:
            print(f"Actual rate: {actual_rate:.2f} Hz, collected frames: {frame_idx}")

        frame_idx += 1

    save_robot_streams(
        session_dir=session_dir,
        tcp_rows=tcp_rows,
        joint_rows=joint_rows,
        ext_wrench_rows=ext_wrench_rows,
        timestamps_host_s=timestamps_host_s,
    )


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
