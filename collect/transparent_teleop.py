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
    ) -> None:
        self.first_sn = first_sn
        self.second_sn = second_sn
        self.robot_pair_idx = robot_pair_idx
        self.lock = Lock()
        self.started = False
        self.engaged = False

        robot_pairs = [(self.first_sn, self.second_sn)]
        lan_ips = [] if network_interface_whitelist is None else list(network_interface_whitelist)
        self.cart_teleop = flexivtdk.TransparentCartesianTeleopLAN(robot_pairs, lan_ips)

    def init(self) -> None:
        """Initialize and start the transparent TDK control process."""
        with self.lock:
            self.cart_teleop.Init()
            self.cart_teleop.Start()
            self.started = True

    def activate(self, activated: bool) -> None:
        """Engage or disengage leader/follower teleoperation."""
        with self.lock:
            if not self.started:
                raise RuntimeError("Transparent teleoperation has not been started")
            self.cart_teleop.Engage(self.robot_pair_idx, activated)
            self.engaged = activated

    def read_states(self):
        """Read leader and follower states in one serialized TDK call."""
        with self.lock:
            return self.cart_teleop.robot_states(self.robot_pair_idx)

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

    def read_slave_robot_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        slave_state = self.read_slave_state()
        tcp_xyz, tcp_quat_xyzw = tdk_pose_to_saved_xyzquat(slave_state.tcp_pose)
        joint_angles = np.asarray(slave_state.q, dtype=np.float64)
        ext_wrench_in_tcp = np.asarray(slave_state.ext_wrench_in_tcp, dtype=np.float64)
        return tcp_xyz, tcp_quat_xyzw, joint_angles, ext_wrench_in_tcp

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
        with self.lock:
            if self.engaged:
                try:
                    self.cart_teleop.Engage(self.robot_pair_idx, False)
                except Exception as exc:
                    logger.warning("Failed to disengage transparent teleop: %s", exc)
                self.engaged = False

            if self.started:
                try:
                    self.cart_teleop.Stop()
                except Exception as exc:
                    logger.warning("Failed to stop transparent teleop: %s", exc)
                finally:
                    self.started = False

    def __enter__(self):
        try:
            self.init()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


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

    def read_robot_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.teleop_pair.read_slave_robot_sample()


CartesianTeleopPair = TransparentCartesianTeleopPair
