#!/usr/bin/env python3
"""Transparent Flexiv Cartesian teleoperation adapter for data collection.

This module uses Flexiv TDK's TransparentCartesianTeleopLAN API.  It keeps the
same small interface that the collector expects from the previous teleoperation
adapter, so data collection, force recording, and shutdown behavior stay in one
place.
"""

import logging
from threading import Lock
from typing import Optional, Sequence, Tuple

import numpy as np

# RDK must be imported before TDK to avoid duplicate pybind type registration
# with the SDK versions deployed on the new collection machine.
import flexivrdk  # noqa: F401
import flexivtdk


logger = logging.getLogger(__name__)

TDK_TCP_POSE_ORDER = "[x, y, z, qw, qx, qy, qz]"
SAVED_TCP_POSE_ORDER = "[x, y, z, qx, qy, qz, qw]"


def tdk_pose_to_saved_xyzquat(tdk_pose: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a TDK TCP pose to the dataset position/quaternion order."""
    pose = np.asarray(tdk_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected a 7-element TCP pose, got shape {pose.shape}")

    xyz = pose[:3].copy()
    quat_xyzw = np.array([pose[4], pose[5], pose[6], pose[3]], dtype=np.float64)
    return xyz, quat_xyzw


class TransparentCartesianTeleopPair:
    """Thread-safe wrapper for one transparent leader/follower robot pair."""

    def __init__(
        self,
        first_sn: str,
        second_sn: str,
        robot_pair_idx: int = 0,
        network_interface_whitelist: Optional[Sequence[str]] = None,
        zero_ft_sensors: bool = True,
    ) -> None:
        self.first_sn = first_sn
        self.second_sn = second_sn
        self.robot_pair_idx = robot_pair_idx
        self.lock = Lock()
        self.started = False
        self.engaged = False
        self.zero_ft_sensors = bool(zero_ft_sensors)
        self.leader_robot = None
        self.follower_robot = None

        robot_pairs = [(self.first_sn, self.second_sn)]
        lan_ips = [] if network_interface_whitelist is None else list(network_interface_whitelist)
        self.cart_teleop = flexivtdk.TransparentCartesianTeleopLAN(robot_pairs, lan_ips)

    def init(self) -> None:
        """Initialize and start the transparent TDK control process."""
        with self.lock:
            zero_mode = (
                flexivtdk.ZeroFTSensor.Enable
                if self.zero_ft_sensors
                else flexivtdk.ZeroFTSensor.Disable
            )
            try:
                self.cart_teleop.Init(True, zero_mode)
                self.cart_teleop.Start()
                self.started = True
                self.leader_robot, self.follower_robot = self.cart_teleop.instances(
                    self.robot_pair_idx
                )
            except BaseException:
                try:
                    self.cart_teleop.Stop()
                except Exception:
                    pass
                self.started = False
                self.leader_robot = None
                self.follower_robot = None
                raise

    def set_wrench_feedback_scale(self, factor: float) -> None:
        """Set follower-to-leader wrench feedback scaling for TDK 1.6."""
        with self.lock:
            if not self.started:
                raise RuntimeError("Transparent teleoperation has not been started")
            self.cart_teleop.SetWrenchFeedbackScalingFactor(
                self.robot_pair_idx,
                factor,
            )
            logger.info("Wrench feedback scaling factor set to %.1f", factor)

    def activate(self, activated: bool) -> None:
        """Engage or disengage leader/follower teleoperation."""
        with self.lock:
            if not self.started:
                raise RuntimeError("Transparent teleoperation has not been started")
            if activated and not self.engaged:
                if self.leader_robot is None or self.follower_robot is None:
                    raise RuntimeError("TDK-owned robot instances are unavailable")
                leader_state = self.leader_robot.states()
                follower_state = self.follower_robot.states()
                self.cart_teleop.SetLeaderNullSpacePosture(self.robot_pair_idx, leader_state.q)
                self.cart_teleop.SetFollowerNullSpacePosture(self.robot_pair_idx, follower_state.q)
                logger.info("Initialized leader and follower null-space postures from current joints")
            self.cart_teleop.Engage(self.robot_pair_idx, activated)
            self.engaged = activated

    def read_states(self):
        """Read states from the RDK instances owned by the TDK pair."""
        with self.lock:
            if not self.started or self.leader_robot is None or self.follower_robot is None:
                raise RuntimeError("Transparent teleoperation has not been started")
            return self.leader_robot.states(), self.follower_robot.states()

    def read_master_state(self):
        return self.read_states()[0]

    def read_slave_state(self):
        return self.read_states()[1]

    def read_slave_tcp_pose_and_joints(self) -> Tuple[np.ndarray, np.ndarray]:
        slave_state = self.read_slave_state()
        return (
            np.asarray(slave_state.tcp_pose, dtype=np.float64),
            np.asarray(slave_state.q, dtype=np.float64),
        )

    def read_slave_saved_xyzquat_and_joints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tcp_pose, joint_angles = self.read_slave_tcp_pose_and_joints()
        tcp_xyz, tcp_quat_xyzw = tdk_pose_to_saved_xyzquat(tcp_pose)
        return tcp_xyz, tcp_quat_xyzw, joint_angles

    def read_slave_robot_sample(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        slave_state = self.read_slave_state()
        tcp_xyz, tcp_quat_xyzw = tdk_pose_to_saved_xyzquat(slave_state.tcp_pose)
        joint_angles = np.asarray(slave_state.q, dtype=np.float64)
        tcp_vel = np.asarray(slave_state.tcp_vel, dtype=np.float64)
        ext_wrench_in_tcp = np.asarray(slave_state.ext_wrench_in_tcp, dtype=np.float64)
        return tcp_xyz, tcp_quat_xyzw, joint_angles, tcp_vel, ext_wrench_in_tcp

    def sync_null_space_postures(self):
        """Compatibility hook; transparent teleop owns null-space behavior."""
        return None

    def fault(self):
        with self.lock:
            return self.cart_teleop.fault(self.robot_pair_idx)

    def any_fault(self) -> bool:
        with self.lock:
            return self.cart_teleop.any_fault()

    def is_stopped(self) -> bool:
        with self.lock:
            return not self.started or self.cart_teleop.stopped(self.robot_pair_idx)

    def clear_fault(self, timeout_sec: int = 30):
        with self.lock:
            return self.cart_teleop.ClearFault(timeout_sec)

    def stop(self) -> None:
        """Best-effort disengage, then stop the transparent TDK process."""
        self._stop(release_native=False)

    def close(self) -> None:
        """Stop teleoperation and immediately release the native TDK handle."""
        self._stop(release_native=True)

    def _stop(self, release_native: bool) -> None:
        with self.lock:
            cart_teleop = self.cart_teleop
            if cart_teleop is None:
                return

            if self.engaged:
                try:
                    cart_teleop.Engage(self.robot_pair_idx, False)
                except Exception as exc:
                    logger.warning("Failed to disengage transparent teleop: %s", exc)
                self.engaged = False

            if self.started:
                try:
                    cart_teleop.Stop()
                except Exception as exc:
                    logger.warning("Failed to stop transparent teleop: %s", exc)
                finally:
                    self.started = False
                    self.leader_robot = None
                    self.follower_robot = None

            if release_native:
                self.cart_teleop = None

    def __enter__(self):
        try:
            self.init()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TeleopSlaveStateReader:
    """State-reader interface consumed by dual_collect_utils.py."""

    def __init__(self, teleop_pair: TransparentCartesianTeleopPair) -> None:
        self.teleop_pair = teleop_pair

    def read_slave_state(self):
        return self.teleop_pair.read_slave_state()

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.teleop_pair.read_slave_tcp_pose_and_joints()

    def read_saved_xyzquat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.teleop_pair.read_slave_saved_xyzquat_and_joints()

    def read_robot_sample(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.teleop_pair.read_slave_robot_sample()


CartesianTeleopPair = TransparentCartesianTeleopPair
