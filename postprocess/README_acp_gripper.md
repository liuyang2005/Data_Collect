# 插销任务：ACP 夹爪与触觉后处理/训练

这套扩展不会修改原 adaptive_compliance_policy 和原
postprocess/run_acp_pipeline.sh。它使用：

- adaptive_compliance_policy_extend
- postprocess/convert_acp_gripper_raw.py
- postprocess/run_acp_gripper_pipeline.sh
- 任务配置 pin_insert_gripper_conv.yaml

## 运行

检查 run_acp_gripper_pipeline.sh 顶部参数后执行：

    cd /home/xense/haptic_exo_teleop_ws/liuyang/Data_Collect
    bash postprocess/run_acp_gripper_pipeline.sh

默认使用 push_0728 的腕部相机 cam_260322274925_wrist，运行 raw、zarr
和标签阶段，不自动开始长时间训练。检查处理结果后，把脚本中的
RUN_TRAIN 改成 true。

也可以单独训练：

    cd /home/xense/haptic_exo_teleop_ws/liuyang/adaptive_compliance_policy_extend/PyriteML
    export PYRITE_DATASET_FOLDERS=/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_processed
    export PYRITE_CHECKPOINT_FOLDERS=/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints
    HYDRA_FULL_ERROR=1 /home/xense/miniconda3/envs/pyrite/bin/accelerate launch train.py \\
      --config-name=train_gripper_conv_workspace \\
      task.dataset_path=/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_processed/push_0728_gripper_acp_v1

## 数据与动作定义

单臂动作从 19 维扩展为 20 维：

    [参考位姿 9, 虚拟目标位姿 9, 刚度 1, 夹爪动作 1]

夹爪动作定义为 0=闭合、1=张开。采集文件中的第 8 个 TCP 字段是
slave_gripper.read() 的宽度反馈，不是控制命令；转换器用宽度滞回生成动作标签：

- 宽度小于等于 0.065 m：闭合；
- 宽度大于等于 0.075 m：张开；
- 中间区间：保持上一状态，避免噪声反复切换。

宽度反馈另存为 gripper_width_0 并作为策略观测。部署时不能把约
0.053 m 的接触后反馈直接当成闭合命令；部署映射需要在下一阶段单独接入夹爪。

## 触觉输入

- tactile/force_torque.npy 保存为 6 维触觉力输入，进入 ACP 的时序力编码器；
- marker_offset.npy 完整保留在 raw episode；
- 训练 zarr 默认保存 6 维 marker 特征：mean_xy、std_xy、max_abs_xy；
- 如需在 zarr 中同时保留完整 marker 场，可给 real_data_processing_gripper.py
  添加 --save-full-marker-offset，但当前策略不会直接使用该大张量。

新标签脚本会先减去每条 episode 开头 200 个力样本的零偏，再计算虚拟目标和
刚度。动作下采样改为 10、时域长度为 16（约 1.5 秒），并启用尾部动作补齐，
以覆盖插入后的松开阶段。夹爪维度默认使用 2 倍扩散损失权重，同时记录独立的
夹爪 MSE 和开/关准确率。
