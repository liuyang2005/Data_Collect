import argparse

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


FIXED_INITIAL_JOINTS_DEG = [0, -40, 0, 90, 0, 40, 45]
FIXED_INITIAL_GRIPPER_WIDTH = 0.08

CUSTOM_HOME_JOINTS_DEG = {
    1: FIXED_INITIAL_JOINTS_DEG.copy(),
    2: FIXED_INITIAL_JOINTS_DEG.copy(),
}


def home_robot(robot_id):
    from r3kit.devices.robot.flexiv.rizon import Rizon

    if robot_id == 1:
        robot_sn = "Rizon4s-063652"
        tool_name = "tool1"
    elif robot_id == 2:
        robot_sn = "Rizon4s-063586"
        tool_name = "hapticexoteleop"
    else:
        raise ValueError("Invalid robot ID")

    print(f"Homing robot {robot_id}: {robot_sn}")
    robot = Rizon(id=robot_sn, gripper=False, name="Rizon4s", tool_name=tool_name)
    target_joints_deg = np.asarray(CUSTOM_HOME_JOINTS_DEG[robot_id], dtype=np.float64)
    print(f"Moving to custom home joints (deg): {target_joints_deg.tolist()}")
    robot.motion_mode("joint")
    robot.joint_move(np.deg2rad(target_joints_deg))
    robot.motion_mode("primitive")
    print("Homing command finished")


def main():
    args = parse_args()
    home_robot(args.robot_id)


if __name__ == "__main__":
    main()
