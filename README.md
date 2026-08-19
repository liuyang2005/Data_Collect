# Data Collect

Flexiv 双臂透明遥操作数据采集工具，用于记录从端机械臂的 RGB-D、TCP 位姿与速度、关节角、外力/力矩、夹爪宽度和双指 Xense 触觉数据。所有设备由同一个采集线程按 10 Hz 顺序读取，每个周期在读取开始前分配一个公共主机时间戳。

## 采集前配置

常用参数统一在 `collect/run_dual_collect.sh` 顶部修改。下面列出真机采集前需要重点检查的配置：

```bash
# Robots and output
FIRST_SN="Rizon4R-062116"
SECOND_SN="Rizon4R-062115"
SAVE_ROOT="/home/xense/haptic_exo_teleop_ws/liuyang/Data/pick_0721"
SESSION_NAME=""

# Aligned collection rate and RealSense hardware stream rate
FPS="10"
CAMERA_DEVICE_FPS="30"

# Follower-to-leader wrench feedback
WRENCH_FEEDBACK_SCALE="0.0"

# Slave Xense gripper and dual-fingertip tactile sensors
USE_GRIPPER="true"
SLAVE_GRIPPER_ID="8a429d6ea337"
USE_TACTILE="true"
TACTILE_LEFT_SENSOR_SN="OG001453"
TACTILE_RIGHT_SENSOR_SN="OG001455"
TACTILE_MAC_ADDR="$SLAVE_GRIPPER_ID"

# Master Angler calibration and slave width mapping
ANGLER_ID="/dev/ttyUSB0"
ANGLER_OPEN_ANGLE="51.68"
ANGLER_CLOSE_ANGLE="16.61"
SLAVE_OPEN_WIDTH="0.0"
SLAVE_CLOSE_WIDTH="0.0"
INITIAL_GRIPPER_WIDTH="0.0"

# Optional LAN whitelist and homing
NETWORK_INTERFACES="192.168.97.10"
HOME_AFTER_RECORDING="true"
HOME_ON_EXIT="true"
HOME_ROBOT_IDS="1,2"
```

关键说明：

- `SAVE_ROOT` 是轨迹保存根目录；`SESSION_NAME=""` 时自动创建 `record_YYYYmmdd_HHMMSS_xxxxxx` 目录。
- `FPS="10"` 控制所有模态的统一逻辑采集频率；`CAMERA_DEVICE_FPS="30"` 只控制 RealSense 硬件流 profile，不改变保存频率。
- `WRENCH_FEEDBACK_SCALE="0.0"` 关闭从臂到主臂的 wrench 反馈，`1.0` 开启反馈。两种条件均保留位姿遥操作和 `ext_wrench_in_tcp` 数据采集。
- `SLAVE_GRIPPER_ID`、左右触觉传感器序列号、相机序列号和机器人序列号均为真机相关配置，启动前必须核对。
- Angler 编码器角度会在 `ANGLER_CLOSE_ANGLE` 到 `ANGLER_OPEN_ANGLE` 之间线性映射为 `SLAVE_CLOSE_WIDTH` 到 `SLAVE_OPEN_WIDTH`；宽度单位为米。
- `INITIAL_GRIPPER_WIDTH` 是程序启动和退出时发送给从端夹爪的目标宽度。详细夹爪与触觉参数见 [collect/README.md](collect/README.md)。

## 启动采集

在机器人控制电脑上进入仓库并激活环境：

```bash
cd /home/xense/haptic_exo_teleop_ws/liuyang/Data_Collect
conda activate foar_arm_env
sh collect/run_dual_collect.sh
```

程序启动后使用键盘控制：

```text
r  激活遥操作
s  暂停遥操作
c  开始记录一条轨迹
v  停止并保存当前轨迹，然后自动复位两台机械臂
q  退出程序
```

推荐操作顺序：

```text
启动程序 -> r 激活遥操作 -> c 开始采集 -> v 保存并自动复位
等待 TDK 重新就绪 -> r 重新激活 -> c 继续采集 -> v 保存并自动复位
s 暂停遥操作 -> q 退出程序
```

每次按 `c` 都会创建一条新轨迹。`HOME_AFTER_RECORDING="true"` 时，按 `v` 会先完整保存数据，再停止 TDK、将两台机械臂复位到固定初始关节角，并重新创建 TDK。相机、Angler、从端夹爪和双指触觉在整个进程中保持连接，不会随每条轨迹重复初始化。复位完成后 TDK 保持未激活，需要再次按 `r` 才能开始下一条操作。

## 固定初始点

主端和从端的固定初始关节角在 `collect/homing.py` 中配置，单位为 degree：

```python
FIXED_INITIAL_JOINTS_DEG = [0, -32, 0, 90, 0, 28, 0]

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

`HOME_AFTER_RECORDING="true"` 控制每条成功保存后的自动复位；`HOME_ON_EXIT="true"` 独立控制程序退出时的复位。两者均使用 `HOME_ROBOT_IDS`，执行前必须确认从当前姿态到固定关节角的运动路径无碰撞。

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
    left/
      marker_offset.npy
      force_torque.npy
      force_norm.npy
      timestamps_host_s.npy
      rectify/*.png
      difference/*.png
      depth/*.png
    right/
      marker_offset.npy
      force_torque.npy
      force_norm.npy
      timestamps_host_s.npy
      rectify/*.png
      difference/*.png
      depth/*.png
  timing/
    cycle_index.npy
    cycle_timestamps_host_s.npy
    scheduled_monotonic_ns.npy
    acquisition_started_monotonic_ns.npy
    acquisition_completed_monotonic_ns.npy
    cycle_duration_s.npy
    deadline_overrun_s.npy
    write_started_host_s.npy
    write_completed_host_s.npy
    source_completed_host_s.npz
  metadata.json
```

主要字段：

- `robot/tcp_pose.npy`：`[x, y, z, qx, qy, qz, qw, gripper_width]`。
- `robot/tcp_vel.npy`：`[vx, vy, vz, wx, wy, wz]`，单位为 `[m/s, rad/s]`。
- `robot/q.npy`：`[q1, ..., q7, gripper_width]`。
- `ext_wrench_in_tcp.npy`：从端 TCP 坐标系下的 `[fx, fy, fz, tx, ty, tz]`。
- `tactile/{left,right}/marker_offset.npy`：每只手指相对各自启动基线的 marker 位移。
- `tactile/{left,right}/force_torque.npy`：每只手指的六维 `ForceResultant` 力/力矩。
- `tactile/{left,right}/force_norm.npy`：SDK 原始 `ForceNorm` 法向力分量场，不是由六维合力计算的标量范数。
- 各相机、`robot`、wrench 和左右 tactile 的 `timestamps_host_s.npy`：同一个采集周期开始时分配的公共主机时间戳，数组逐元素完全一致。
- `timing/write_completed_host_s.npy`：writer 完成该周期全部 PNG 写入并接收数值载荷的应用层完成时间；数值 `.npy` 在停止录制时统一生成。
- `timing/source_completed_host_s.npz`：各设备实际读取完成时间，仅用于诊断顺序读取偏差，不参与数据对齐。
- `metadata.json`：本次采集参数、设备信息、采样频率和数据格式说明。

每个完整周期只提交一次，因此所有启用模态的行数、图像编号和公共时间戳一一对应，可以直接按数组行号使用，不需要最近邻、插值或额外时间对齐。设备接口仍然是顺序调用，公共时间戳表示同一逻辑周期，不表示相机和左右触觉严格同时曝光。

## Tools

`tools/visualize_episode.py` 用于生成单条轨迹的时序、采样间隔和图像概览，`tools/align_datacollect_timestamps.py` 用于检查相机、机器人、wrench 与触觉时间戳对齐情况。具体命令和输出文件见 [tools/README.md](tools/README.md)。

## 真机注意事项

- 启动前确认两台机械臂无人占用、急停可用，并处于 TDK 要求的远程控制状态。
- 核对机器人序列号、Xense ID、触觉传感器序列号、相机连接和保存路径。
- TDK 初始化期间保持机械臂末端无外部接触，完成初始化后再按 `r` 激活遥操作。
- 不要在机械臂被他人使用时运行 `collect/run_dual_collect.sh` 或 `collect/homing.py`。
- 执行 homing 前确认运动路径和目标关节角安全；异常退出后先检查机器人状态，再决定是否复原。
- `HOME_ON_EXIT="true"` 只在采集程序完成初始化并进入键盘控制后生效。
