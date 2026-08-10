import argparse
import time

import numpy as np


"""
Usage:
    python homing.py -id <1 or 2>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose which robot to home",
    )
    parser.add_argument(
        "-id",
        "--id",
        dest="robot_id",
        type=int,
        choices=[1, 2],
        required=True,
        help="1 refers to master robot, 2 refers to slave robot",
    )
    return parser.parse_args()


FIXED_INITIAL_JOINTS_DEG = [0, -32, 0, 90, 0, 28, 0]
FIXED_INITIAL_GRIPPER_WIDTH = 0.08

CUSTOM_HOME_JOINTS_DEG = {
    1: FIXED_INITIAL_JOINTS_DEG.copy(),
    2: FIXED_INITIAL_JOINTS_DEG.copy(),
}

ROBOT_CONFIGS = {
    1: {
        "serial": "Rizon4R-062116",
        "tool": "tool1",
        "local_ips": ["192.168.97.10"],
    },
    2: {
        "serial": "Rizon4R-062115",
        "tool": "xense",
        "local_ips": ["192.168.97.10"],
    },
}


def home_robot(
    robot_id,
    enable_timeout_s=10.0,
    move_timeout_s=5.0,
    joint_tolerance_rad=0.02,
    wait_interval_s=0.01,
):
    import flexivrdk

    if robot_id not in ROBOT_CONFIGS:
        raise ValueError("Invalid robot ID")

    config = ROBOT_CONFIGS[robot_id]
    robot_sn = config["serial"]
    tool_name = config["tool"]

    print(f"Homing robot {robot_id}: {robot_sn}")
    robot = flexivrdk.Robot(robot_sn, config["local_ips"])
    try:
        if not robot.estop_released():
            raise RuntimeError(f"E-stop is pressed for {robot_sn}")
        if robot.fault() and not robot.ClearFault():
            raise RuntimeError(f"Failed to clear fault for {robot_sn}")
        if not robot.operational():
            robot.Enable()
            deadline = time.monotonic() + enable_timeout_s
            while not robot.operational():
                if robot.fault():
                    raise RuntimeError(f"{robot_sn} entered fault while enabling")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Failed to enable {robot_sn}")
                time.sleep(wait_interval_s)

        robot.SwitchMode(flexivrdk.Mode.IDLE)
        tool = flexivrdk.Tool(robot)
        if not tool.exist(tool_name):
            raise RuntimeError(f"{tool_name} tool not found on {robot_sn}")
        tool.Switch(tool_name)

        target_joints_deg = np.asarray(
            CUSTOM_HOME_JOINTS_DEG[robot_id], dtype=np.float64
        )
        target_joints = np.deg2rad(target_joints_deg)
        info = robot.info()
        target_joints = np.clip(
            target_joints,
            np.asarray(info.q_min, dtype=np.float64),
            np.asarray(info.q_max, dtype=np.float64),
        )
        print(f"Moving to custom home joints (deg): {target_joints_deg.tolist()}")
        robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
        zeros = np.zeros(7, dtype=np.float64).tolist()
        limits = np.ones(7, dtype=np.float64).tolist()
        robot.SendJointPosition(
            target_joints.tolist(),
            zeros,
            limits,
            limits,
        )

        deadline = time.monotonic() + move_timeout_s
        while True:
            current_joints = np.asarray(robot.states().q, dtype=np.float64)
            if np.max(np.abs(current_joints - target_joints)) <= joint_tolerance_rad:
                break
            if robot.fault():
                raise RuntimeError(f"{robot_sn} entered fault while moving home")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Custom home move timed out for {robot_sn}")
            time.sleep(wait_interval_s)

        robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        print("Homing command finished")
    finally:
        robot.Stop()


def main():
    args = parse_args()
    home_robot(args.robot_id)


if __name__ == "__main__":
    main()
