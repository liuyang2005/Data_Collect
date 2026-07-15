# Data Collect

Flexiv 双臂遥操作数据采集工具，用于采集从端机械臂的 RGB-D、TCP 位姿、关节角、外力/力矩和夹爪宽度，并提供面向 ACP 与 MaskACT-3D 的后处理脚本。

当前遥操作层使用 Flexiv TDK 的 `TransparentCartesianTeleopLAN` 接口；采集层保留相机、机器人状态、力数据的多频率保存逻辑。

## 快速运行

在机器人控制电脑上进入仓库并激活环境：

```bash
cd /home/xense/haptic_exo_teleop_ws/jiaqingke/Data_Collect
conda activate foar_arm_env
sh collect/run_dual_collect.sh
```

默认运行参数在 `collect/run_dual_collect.sh` 顶部修改：

```bash
FIRST_SN="Rizon4s-063652"
SECOND_SN="Rizon4s-063586"
SAVE_ROOT="/home/xense/haptic_exo_teleop_ws/jiaqingke/Data/pick_0713"
FPS="30"
CAMERA_FPS="30"
ROBOT_FPS="100"
FORCE_FPS="200"
USE_GRIPPER="true"
HOME_ON_EXIT="true"
HOME_ROBOT_IDS="1,2"
INITIAL_GRIPPER_WIDTH="0.08"
```

运行后键盘控制：

```text
r  激活遥操作
s  暂停遥操作
c  开始记录一条轨迹
v  结束当前轨迹记录
q  退出程序
```

推荐流程：

```text
启动程序 -> r 激活遥操作 -> c 开始采集 -> v 停止当前轨迹
回到下一条轨迹起点 -> c 继续采集 -> v 停止
s 暂停遥操作 -> q 退出程序
```

如果 `HOME_ON_EXIT=true`，程序在完成初始化并进入键盘控制后退出时，会让 `HOME_ROBOT_IDS` 指定的机械臂回到固定初始关节角。若初始化未完成，例如相机或 TDK 初始化失败，则不会触发退出复原。

## 固定初始点

主端和从端共用同一个固定初始点，在 `collect/homing.py` 中配置，单位为 degree：

```python
FIXED_INITIAL_JOINTS_DEG = [0.87, 0.71, 6.22, 107.67, 5.33, 20.44, 50.42]

CUSTOM_HOME_JOINTS_DEG = {
    1: FIXED_INITIAL_JOINTS_DEG.copy(),
    2: FIXED_INITIAL_JOINTS_DEG.copy(),
}
```

也可以单独复原某一台机械臂：

```bash
python collect/homing.py -id 1
python collect/homing.py -id 2
```

## 采集数据结构

每次按 `c` 开始记录时，会在 `SAVE_ROOT` 下创建一条 `record_YYYYmmdd_HHMMSS_xxxxxx` 轨迹目录。典型结构：

```text
record_YYYYmmdd_HHMMSS_xxxxxx/
  cam_327322062498/
    color/
      0000000000000000.png
    depth/
      0000000000000000.png
    timestamps_host_s.npy
  cam_260322274925_wrist/
    color/
      0000000000000000.png
    depth/
      0000000000000000.png
    timestamps_host_s.npy
  robot/
    tcp_pose.npy
    tcp_vel.npy
    q.npy
    timestamps_host_s.npy
  ext_wrench_in_tcp.npy
  ext_wrench_in_tcp_timestamps_host_s.npy
  metadata.json
```

主要字段：

- `robot/tcp_pose.npy`：从端 TCP，格式为 `[x, y, z, qx, qy, qz, qw, gripper_width]`。
- `robot/tcp_vel.npy`：从端相对 world frame 的 TCP 速度，格式为 `[vx, vy, vz, wx, wy, wz]`，单位为 `[m/s, rad/s]`。
- `robot/q.npy`：从端 7 关节角加夹爪宽度，格式为 `[q1, ..., q7, gripper_width]`。
- `robot/timestamps_host_s.npy`：上述三个 robot 数组共用的主机时间戳。
- `ext_wrench_in_tcp.npy`：从端 TCP 坐标系下外力/力矩，格式为 `[fx, fy, fz, tx, ty, tz]`。
- `ext_wrench_in_tcp_timestamps_host_s.npy`：独立 wrench 流的主机时间戳。
- `metadata.json`：采集参数、相机序列号、采样频率和位姿格式说明。
- 默认相机目录：`cam_327322062498` 为主视角，`cam_260322274925_wrist` 为腕部相机。

## 代码目录

```text
collect/
  dual_collect.py          数据采集主入口
  transparent_teleop.py    Flexiv Transparent TDK 遥操作封装
  dual_collect_utils.py    相机、夹爪、目录和多频率数据保存工具
  homing.py                固定初始点复原脚本
  run_dual_collect.sh      常用启动脚本
  teleop_health_check.py   遥操作健康检查脚本

calib/
  相机内参、外参采集和标定工具

postprocess/
  convert_acp_raw.py       转换为 ACP raw 数据格式
  run_acp_pipeline.sh      ACP raw -> zarr -> 可选训练的一键脚本
  convert_hdf5.py          转换为 MaskACT-3D HDF5
  run_convert_hdf5.sh      HDF5 转换启动脚本

validate_data/
  visualize_hdf5_pointcloud.py  HDF5 点云可视化与单帧诊断
  replay_hdf5_tcp_pybullet.py   TCP 轨迹 PyBullet 回放
  run_validate_hdf5.sh          验证启动脚本
```

各子目录内还有更详细的 README：

```text
collect/README.md
postprocess/README.md
validate_data/README.md
```

## 后处理

### ACP raw / zarr

采集完成后，先修改 `postprocess/run_acp_pipeline.sh` 顶部路径和数据集名，然后运行：

```bash
sh postprocess/run_acp_pipeline.sh
```

如果只需要转换 raw 输入格式：

```bash
sh postprocess/run_convert_acp_raw.sh
```

### MaskACT-3D HDF5

先修改 `postprocess/run_convert_hdf5.sh` 顶部参数：

```bash
SESSIONS="/path/to/save_root_or_record"
OUTPUT_HDF5="/path/to/train_data.hdf5"
CAMERA_NAME="cam_327322062498"
```

然后运行：

```bash
sh postprocess/run_convert_hdf5.sh
```

### 数据验证

HDF5 生成后可用：

```bash
sh validate_data/run_validate_hdf5.sh
```

或直接运行点云 summary 检查：

```bash
python validate_data/visualize_hdf5_pointcloud.py \
  --hdf5 /path/to/train_data.hdf5 \
  --summary-only
```

## 注意事项

- 启动采集前确认两台机械臂无人占用、急停和远程模式状态正常。
- 不要在机械臂被他人使用时运行 `collect/run_dual_collect.sh` 或 `collect/homing.py`。
- `HOME_ON_EXIT=true` 只在程序完成初始化并进入键盘控制后生效。
- RealSense 相机默认采集 `cam_327322062498` 主视角和 `cam_260322274925_wrist` 腕部相机；后处理脚本一次导出一个相机视角，需要在对应脚本中设置 `CAMERA_NAME`。
- 公开仓库前建议清理真实设备序列号、Xense ID、本机路径和实验数据路径。

## 参考

- Flexiv TDK Python examples: https://github.com/flexivrobotics/flexiv_tdk/tree/main/example_py
- Transparent Cartesian Teleoperation LAN example: https://github.com/flexivrobotics/flexiv_tdk/blob/main/example_py/transparent_cartesian_teleop_lan.py
