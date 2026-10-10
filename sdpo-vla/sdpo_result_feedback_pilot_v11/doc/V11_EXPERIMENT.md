# V11：方向性特权信号的 no-op 基线残差

## V10 审计结论

V10 的 `sdpo_directional` 教师直接使用带目标方向的教师分布：

```text
teacher = pi_teacher(image, proprio, result + target_direction)
student_target = teacher
```

它没有计算同一教师、同一图像/状态、同一结果下把方向建议替换为 `E=hold` 的分布。因此，教师因特权方向上下文产生的整体方向偏置会直接进入学生目标。V10 的 `teacher_log_probs` 只保存最终教师分布，不能事后恢复该偏置。

## V11 修正

只对 directional 条件启用：

```text
privileged = pi_teacher(image, proprio, result + measured_direction)
baseline = pi_teacher(image, proprio, result + zero_direction/hold)
residual = log(privileged) - log(baseline)
target = softmax(log(student) + residual)
```

`baseline` 与 `privileged` 使用相同 EMA 教师、图像、proprio、任务、动作结果；只将 `target_dx_m`、`target_dy_m` 和 `target_direction` 改为 `0, 0, hold`。`E` 是基线语义，不是额外专家动作标签。

普通常数 logit 偏移在差分中消失。真正改变动作相对偏好的方向信号保留，并以相对当前学生的残差形式蒸馏。

## 验收

- `apply_teacher_residual` 单测证明共同 logit 偏移被抵消。
- `no_op_directional_feedback` 单测证明结果字段不变，只替换方向字段。
- 每个 V11 update 保存 `teacher_privileged_log_probs`、`teacher_noop_baseline_log_probs`、`teacher_residual_log_probs`。
- `subtract_noop_baseline=true` 只出现在 residual 条件。
- V10 原始目录保持不变，作为 raw 对照。
