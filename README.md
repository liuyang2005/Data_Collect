# Data Collect

Flexiv 双臂透明遥操作数据采集工具，用于记录从端机械臂的 RGB-D、TCP 位姿与速度、关节角、外力/力矩、夹爪宽度和单指 Xense 触觉数据。相机、机器人状态、wrench 和触觉使用独立采样频率，并分别保存主机时间戳。

## 采集前配置

常用参数统一在 `collect/run_dual_collect.sh` 顶部修改。下面列出真机采集前需要重点检查的配置：

```bash
# Robots and output
FIRST_SN="Rizon4s-063652"
SECOND_SN="Rizon4s-063586"
SAVE_ROOT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/pick_0721"
SESSION_NAME=""

# Stream rates
FPS="30"
CAMERA_FPS="30"
ROBOT_FPS="100"
FORCE_FPS="200"

# Follower-to-leader wrench feedback
WRENCH_FEEDBACK_SCALE="0.0"

# Slave Xense gripper and left-fingertip tactile sensor
USE_GRIPPER="true"
SLAVE_GRIPPER_ID="d254505bfaaa"
USE_TACTILE="true"
TACTILE_FPS="60"
TACTILE_SENSOR_SN="OG000451"
TACTILE_MAC_ADDR="$SLAVE_GRIPPER_ID"

# Master Angler calibration and slave width mapping
ANGLER_ID="/dev/ttyUSB0"
ANGLER_OPEN_ANGLE="51.68"
ANGLER_CLOSE_ANGLE="16.61"
SLAVE_OPEN_WIDTH="0.0"
SLAVE_CLOSE_WIDTH="0.0"
INITIAL_GRIPPER_WIDTH="0.0"

# Optional LAN whitelist and exit homing
NETWORK_INTERFACES=""
HOME_ON_EXIT="true"
HOME_ROBOT_IDS="1,2"
```

关键说明：

- `SAVE_ROOT` 是轨迹保存根目录；`SESSION_NAME=""` 时自动创建 `record_YYYYmmdd_HHMMSS_xxxxxx` 目录。
- `CAMERA_FPS`、`ROBOT_FPS`、`FORCE_FPS` 和 `TACTILE_FPS` 分别控制四类独立数据流。
- `WRENCH_FEEDBACK_SCALE="0.0"` 关闭从臂到主臂的 wrench 反馈，`1.0` 开启反馈。两种条件均保留位姿遥操作和 `ext_wrench_in_tcp` 数据采集。
- `SLAVE_GRIPPER_ID`、`TACTILE_SENSOR_SN`、相机序列号和机器人序列号均为真机相关配置，启动前必须核对。
- Angler 编码器角度会在 `ANGLER_CLOSE_ANGLE` 到 `ANGLER_OPEN_ANGLE` 之间线性映射为 `SLAVE_CLOSE_WIDTH` 到 `SLAVE_OPEN_WIDTH`；宽度单位为米。
- `INITIAL_GRIPPER_WIDTH` 是程序启动和退出时发送给从端夹爪的目标宽度。详细夹爪与触觉参数见 [collect/README.md](collect/README.md)。

## 启动采集

在机器人控制电脑上进入仓库并激活环境：

```bash
cd /home/xense/haptic_exo_teleop_ws/jiaqingke/Data_Collect
conda activate foar_arm_env
sh collect/run_dual_collect.sh
```

程序启动后使用键盘控制：

```text
r  激活遥操作
s  暂停遥操作
c  开始记录一条轨迹
v  停止当前轨迹记录
q  退出程序
```

推荐操作顺序：

```text
启动程序 -> r 激活遥操作 -> c 开始采集 -> v 停止当前轨迹
回到下一条轨迹起点 -> c 继续采集 -> v 停止
s 暂停遥操作 -> q 退出程序
```

每次按 `c` 都会创建一条新轨迹。正常完成一条轨迹后，可通过遥操作返回下一条轨迹的起点，不需要重复执行 homing。

## 固定初始点

主端和从端的固定初始关节角在 `collect/homing.py` 中配置，单位为 degree：

```python
FIXED_INITIAL_JOINTS_DEG = [0.87, 0.71, 6.22, 107.67, 5.33, 20.44, 50.42]

CUSTOM_HOME_JOINTS_DEG = {
    1: FIXED_INITIAL_JOINTS_DEG.copy(),
    2: FIXED_INITIAL_JOINTS_DEG.copy(),
}
```

单独复原某一台机械臂：

```bash
python3 collect/homing.py -id 1
python3 collect/homing.py -id 2
```

当 `HOME_ON_EXIT="true"` 时，程序完成初始化并进入键盘控制后退出，会复原 `HOME_ROBOT_IDS` 指定的机械臂；初始化失败时不会自动复原。

## 采集数据

典型轨迹目录如下。相机目录名来自实际设备序列号，不应依赖 README 中的固定编号：

```text
record_YYYYmmdd_HHMMSS_xxxxxx/
  cam_<serial>/
    color/*.png
    depth/*.png
    timestamps_host_s.npy
  cam_<serial>_wrist/
    color/*.png
    depth/*.png
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
    rectify/*.png
    difference/*.png
    depth/*.png
  metadata.json
```

主要字段：

- `robot/tcp_pose.npy`：`[x, y, z, qx, qy, qz, qw, gripper_width]`。
- `robot/tcp_vel.npy`：`[vx, vy, vz, wx, wy, wz]`，单位为 `[m/s, rad/s]`。
- `robot/q.npy`：`[q1, ..., q7, gripper_width]`。
- `ext_wrench_in_tcp.npy`：从端 TCP 坐标系下的 `[fx, fy, fz, tx, ty, tz]`。
- `tactile/force_torque.npy`：左指 Xense 的六维力/力矩。
- `metadata.json`：本次采集参数、设备信息、采样频率和数据格式说明。

不同数据流的行数通常不同，不能按数组行号直接对应。后续使用数据时应依据各自的 `timestamps_host_s.npy` 做最近邻、插值或窗口对齐。

## Tools

`tools/visualize_episode.py` 用于生成单条轨迹的时序、采样间隔和图像概览，`tools/align_datacollect_timestamps.py` 用于检查相机、机器人、wrench 与触觉时间戳对齐情况。具体命令和输出文件见 [tools/README.md](tools/README.md)。

## 真机注意事项

- 启动前确认两台机械臂无人占用、急停可用，并处于 TDK 要求的远程控制状态。
- 核对机器人序列号、Xense ID、触觉传感器序列号、相机连接和保存路径。
- TDK 初始化期间保持机械臂末端无外部接触，完成初始化后再按 `r` 激活遥操作。
- 不要在机械臂被他人使用时运行 `collect/run_dual_collect.sh` 或 `collect/homing.py`。
- 执行 homing 前确认运动路径和目标关节角安全；异常退出后先检查机器人状态，再决定是否复原。
- `HOME_ON_EXIT="true"` 只在采集程序完成初始化并进入键盘控制后生效。
