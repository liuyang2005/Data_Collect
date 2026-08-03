# 双臂遥操作数据采集使用说明

## 文件说明

- `dual_collect.py`：数据采集主入口。
- `transparent_teleop.py`：Flexiv TDK `TransparentCartesianTeleopLAN` 遥操作封装。
- `dual_teleop.py`：旧版 `CartesianTeleopLAN` 封装，保留作参考。
- `dual_collect_utils.py`：相机、夹爪、目录创建和多频率数据保存工具，默认采集主视角相机和腕部相机。
- `homing.py`：固定初始关节角复原脚本。

## 基本用法

在机器人运行环境中执行：

```bash
sh collect/run_dual_collect.sh
```

常用参数可以直接在 `run_dual_collect.sh` 顶部修改。

也可以直接用命令行执行：

```bash
python collect/dual_collect.py \
  -1 <master_robot_sn> \
  -2 <slave_robot_sn> \
  --slave-gripper-id <slave_xense_id> \
  --use-tactile true \
  --tactile-fps 60 \
  --tactile-sensor-sn OG000451 \
  --tactile-mac-addr <slave_xense_id> \
  --save-root <save_root>
```

其中：

- `-1, --first-sn`：主臂序列号。
- `-2, --second-sn`：从臂序列号。
- `--slave-gripper-id`：从端 Xense 夹爪 ID。
- `--save-root`：数据保存根目录。

## 不采集夹爪

默认会初始化和采集夹爪。如果本次不需要夹爪：

```bash
python collect/dual_collect.py \
  -1 <master_robot_sn> \
  -2 <slave_robot_sn> \
  --save-root <save_root> \
  --use-gripper false
```

此时不会初始化 Xense 和 Angler，保存的夹爪宽度固定为 `0.0`。

## 常用可选参数

```bash
--fps 30
--camera-fps 30
--robot-fps 100
--force-fps 200
--wrench-feedback-scale 0.0
--tactile-fps 60
--session-name record_test
--network-interface 192.168.2.102
--gripper-eps 0.0001
--gripper-wait-time 0.1
--null-space-period 0.1
--home-on-exit true
--home-robot-ids 1,2
--home-delay 0.5
--initial-gripper-width 0.08
```

`--fps` 是兼容旧脚本的默认频率；如果没有显式传入 `--camera-fps` 或 `--robot-fps`，对应数据流会回退使用 `--fps`；如果没有显式传入 `--force-fps`，力数据会回退使用 `--robot-fps`，再回退到 `--fps`。当前多频版本中：

- `--camera-fps`：RGBD 相机采集线程频率。
- `--robot-fps`：从臂 TCP、关节角和夹爪宽度采集线程频率。
- `--force-fps`：从臂外力估计 `ext_wrench_in_tcp` 采集线程频率。
- `--wrench-feedback-scale`：TDK 从臂到主臂的 wrench 反馈比例；`0.0` 关闭，`1.0` 开启。关闭反馈不影响位姿遥操作和 wrench 数据采集。
- `--tactile-fps`：左指 Xense 触觉采集线程频率；与相机、机器人和腕力线程相互独立。

启用 `--use-tactile true` 时，需要同时提供夹爪 MAC 地址。默认左指序列号为 `OG000451`：

```bash
--use-tactile true
--tactile-sensor-sn OG000451
--tactile-mac-addr d254505bfaaa
```

`--network-interface` 可以重复传入多个 LAN 网卡 IPv4 地址。

`--home-on-exit true` 时，程序在完成初始化并进入键盘控制后退出，会调用 `homing.py` 将指定机械臂复原到固定初始关节角；如果初始化未完成，则不会触发退出复原。

## 键盘控制

程序启动后：

- `r`：激活主从遥操作。
- `s`：暂停主从遥操作。
- `c`：开始记录一条新轨迹。
- `v`：结束当前轨迹记录。
- `q`：退出采集。

推荐流程：

```text
启动程序 -> r 启动遥操作 -> c 开始记录 -> v 结束记录
移动机械臂回到起点 -> c 记录下一条 -> v 结束下一条
s 暂停遥操作 -> q 退出程序
```

每次按 `c` 都会创建一个新的轨迹目录。当前版本会启动独立采集线程：相机按 `--camera-fps` 采集 RGBD，机器人状态按 `--robot-fps` 保存从臂 TCP、从臂关节角和从端夹爪宽度，力数据按 `--force-fps` 保存外力估计，左指触觉按 `--tactile-fps` 采集完整帧。相机和触觉取帧分别使用独立的 PNG writer，避免 `cv2.imwrite()` 直接拖慢采样线程。

## 数据结构

每次运行会在 `save_root` 下创建一个 session 目录：

```text
record_YYYYmmdd_HHMMSS/
  cam_327322062498/
    color/
    depth/
    timestamps_host_s.npy
  cam_260322274925_wrist/
    color/
    depth/
    timestamps_host_s.npy
  robot/
    tcp_pose.npy
    tcp_vel.npy
    q.npy
    timestamps_host_s.npy
  ext_wrench_in_tcp.npy
  ext_wrench_in_tcp_timestamps_host_s.npy
  tactile/
    marker_offset.npy
    force_torque.npy
    timestamps_host_s.npy
    rectify/
    difference/
    depth/
  metadata.json
```

保存格式：

- `cam_327322062498/color/*.png`：主视角 RGB 图像，文件名为相机线程内连续帧号。
- `cam_260322274925_wrist/color/*.png`：腕部相机 RGB 图像。
- `cam_*/color/*.png`：任意相机 RGB 图像通配路径。
- `cam_*/depth/*.png`：对应相机的 depth 图像，文件名与同一相机 color 帧号一致。
- `cam_*/timestamps_host_s.npy`：`(T_camera,)`，相机帧保存时的主机时间戳，单位秒。
- `robot/tcp_pose.npy`：`(T_robot, 8)`，每行 `[x, y, z, qx, qy, qz, qw, gripper_width]`
- `robot/tcp_vel.npy`：`(T_robot, 6)`，每行来自从臂 `RobotStates.tcp_vel`，顺序为 `[vx, vy, vz, wx, wy, wz]`，相对 world frame，单位为 `[m/s, rad/s]`
- `robot/q.npy`：`(T_robot, 8)`，每行 `[q1, q2, q3, q4, q5, q6, q7, gripper_width]`
- `robot/timestamps_host_s.npy`：`(T_robot,)`，上述三个 robot 数组共用的主机时间戳，单位秒
- `ext_wrench_in_tcp.npy`：`(T, 6)`，每行来自从臂 `RobotStates.ext_wrench_in_tcp`
- `ext_wrench_in_tcp_timestamps_host_s.npy`：`(T_force,)`，力数据采样时的主机时间戳，单位秒
- `tactile/marker_offset.npy`：`(T_tactile, ...)`，`float32`，左指 marker 点阵相对启动基线的偏移
- `tactile/force_torque.npy`：`(T_tactile, 6)`，`float64`，左指 `[Fx, Fy, Fz, Tx, Ty, Tz]`，保留 Xense SDK 原始单位
- `tactile/timestamps_host_s.npy`：`(T_tactile,)`，触觉帧读取后的主机时间戳，单位秒
- `tactile/{rectify,difference,depth}/*.png`：同一触觉行对应的三类图像，使用六位连续编号

`tcp_pose`、`tcp_vel` 和 `q` 从同一次从臂 `RobotStates` 快照提取，所以三者逐行对应并共享时间戳。多频版本中，`T_camera`、`T_robot`、`T_force` 和 `T_tactile` 通常不同，不能假设图片帧号、robot 行号、wrench 行号与触觉行号一一对应；后续 ACP 或 LeRobot 转换脚本应按各自时间戳做最近邻、插值或窗口聚合对齐。

## 主端 Angler 编码器控制夹爪

主端使用 Angler 编码器控制装置，可以在 `run_dual_collect.sh` 中设置：

```bash
USE_GRIPPER="true"
ANGLER_ID="/dev/ttyUSB0"
ANGLER_INDEX="1"
ANGLER_BAUDRATE="1000000"
ANGLER_GAP="-1"
ANGLER_STRICT="true"
ANGLER_OPEN_ANGLE="51.68"
ANGLER_CLOSE_ANGLE="16.61"
SLAVE_OPEN_WIDTH="0.085"
SLAVE_CLOSE_WIDTH="0.0"
```

编码器角度会被线性映射为从端夹爪目标宽度：

```text
ANGLER_CLOSE_ANGLE -> SLAVE_CLOSE_WIDTH
ANGLER_OPEN_ANGLE  -> SLAVE_OPEN_WIDTH
```

从端仍然使用 Xense，采集保存的 `gripper_width` 仍然来自 `slave_gripper.read()`。
