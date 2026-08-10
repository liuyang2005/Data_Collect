# 插销任务 v2：从宽度反馈推断夹爪命令

推荐入口：

    cd /home/xense/haptic_exo_teleop_ws/liuyang/Data_Collect
    bash postprocess/run_acp_gripper_command_single_cycle_pipeline.sh

输出使用新名字，不覆盖 v1：

    Data/acp_raw/push_0728_gripper_command_acp_v2
    Data/acp_processed/push_0728_gripper_command_acp_v2

采集第 8 维是 `slave_gripper.read()` 的宽度反馈。v2 先以 `0.065/0.075 m`
找到可靠状态穿越，再向前搜索持续运动起点，并额外补偿 `0.05 s` 的命令到反馈
延迟。宽度保留为观测，动作第 20 维使用提前后的命令状态，语义仍是
`0=闭合、1=张开`。

默认检测参数：

- 速度阈值：`0.002 m/s`；
- 允许运动间隙：`0.08 s`；
- 最大回看：`0.75 s`；
- 最小宽度变化：`0.003 m`；
- 中值平滑：5 个机器人采样点。

`push_0728` 的 99 条示教中，97 条是单次闭合再张开；2 条包含重抓形成两轮开闭。
推荐流水线会跳过后两条，使训练标签与部署的单循环状态机一致。每条转换细节位于
episode 的 `conversion_metadata.json`，总览位于 `gripper_command_summary.json`。

训练时继续使用 `train_gripper_conv_workspace`，但必须覆盖 v2 路径：

    task.dataset_path=/home/xense/haptic_exo_teleop_ws/liuyang/Data/acp_processed/push_0728_gripper_command_acp_v2

后续新采集应直接保存 `master_gripper.read()`、实际发送给从端的目标宽度及其时间戳，
避免再次通过反馈推断命令。
