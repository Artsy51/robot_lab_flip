# StructureCommand 构型切换训练设计

本文档记录构型命令控制器在训练过程中实现构型切换的完整设计，包括构型加载、命令采样、reset、平滑插值、观测、奖励和训练阶段。

## 1. 目标

训练一个 policy，使机器人能够在运动过程中从当前构型切换到目标构型，并在切换前后保持稳定行走。

需要区分两种模式：

### 环境强制切换

命令控制器直接调用 `write_joint_state_to_sim` 写入关节位置。机器人会被环境强制切换，适合验证构型表、command 和插值逻辑，但 policy 不会真正学会切换动作。

### Policy 学习切换

命令控制器只生成目标构型和参考轨迹，policy 通过关节位置或力矩动作跟踪参考轨迹。奖励评价 policy 是否平稳完成切换。这才是最终的构型切换训练目标。

推荐先用环境强制切换验证系统，再逐步切换到 policy 执行模式。

## 2. 统一职责

`StructureCommand` 应统一负责：

```text
读取 structure_list
    -> 根据关节名正则解析为 tensor
    -> 保存 structure_id command
    -> reset 时随机选择构型
    -> 检测运行中的目标构型变化
    -> 生成平滑插值参考
    -> 提供切换进度和目标位置观测
```

不要再让 `StructureResetManager`、reset event 和 command controller 分别维护不同的构型状态。最终建议将构型状态集中在 `StructureCommand` 中，旧事件只作为兼容或逐步移除。

## 3. 构型列表加载

配置阶段保存字典列表，不访问机器人 asset：

```python
structure_list = [
    FERE_POS,
    FERK_POS,
    FKRE_POS,
    FKRK_POS,
]
```

环境和机器人创建后，在 `StructureCommand.__init__` 中解析：

```python
for structure in cfg.structure_list:
    joint_pos = default_joint_pos.clone()
    for joint_pattern, position in structure.items():
        joint_ids, _ = asset.find_joints(joint_pattern)
        joint_pos[joint_ids] = position
    structure_positions.append(joint_pos)
```

最终保存：

```python
structure_positions.shape == [num_structures, num_selected_joints]
```

每个结构 ID 对应一个绝对关节位置 tensor：

```python
structure_positions[0]  # 结构 0
structure_positions[1]  # 结构 1
```

必须确认四个结构字典确实不同，否则虽然 command ID 不同，机器人实际姿态仍然相同。

## 4. Command 采样时机

### Reset 采样

如果目标是每个 episode 使用一个固定构型：

```text
episode reset
    -> 随机采样 structure_id
    -> 设置对应初始关节位置
    -> episode 内保持不变
```

此时可以使用很大的重采样时间：

```python
resampling_time_range=(1.0e9, 1.0e9)
```

### 运行中切换

如果目标是训练运动中切换，则允许 command 周期变化，例如：

```python
resampling_time_range=(5.0, 5.0)
```

每次 command 从一个结构 ID 变为另一个结构 ID 时，启动一次 transition。

切换过程中建议禁止再次切换，或者将新请求放入队列，等当前 transition 完成后再处理。

## 5. 每个环境独立保存状态

Isaac Lab 是向量化环境，不同环境可能在不同时间 reset 或切换。因此不能使用一个全局 `alpha`。

每个环境至少需要保存：

```python
previous_joint_pos       # 切换开始时的位置
current_joint_pos        # 当前插值参考位置
 target_joint_pos        # 最终目标构型位置
transition_alpha         # [0, 1] 的切换进度
last_structure_id        # 上一次目标结构 ID
transition_active        # 是否正在切换
previous_transition      # 用于只发放一次完成奖励
```

建议 tensor 形状：

```python
[num_envs, num_joints]
[num_envs]
```

## 6. 平滑插值

检测到 command 改变时：

```python
previous_joint_pos[changed_ids] = current_joint_pos[changed_ids]
target_joint_pos[changed_ids] = structure_positions[new_ids]
transition_alpha[changed_ids] = 0.0
transition_active[changed_ids] = True
```

每个 environment step 或 physics step 推进：

```python
alpha += dt / transition_duration
alpha = clamp(alpha, 0.0, 1.0)
```

线性插值：

```python
joint_pos = (
    (1.0 - alpha) * previous_joint_pos
    + alpha * target_joint_pos
)
```

推荐使用三次 smoothstep：

```python
smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
```

如果构型差异较大，推荐使用五次 smootherstep：

```python
smooth_alpha = (
    10.0 * alpha**3
    - 15.0 * alpha**4
    + 6.0 * alpha**5
)
```

五次插值在起点和终点具有更平滑的速度和加速度。

## 7. 防止构型差异导致突变

构型 0 和构型 1 差异较大时，风险不只是关节位置变化，也包括观测和奖励目标变化。

### 延长切换时间

建议初始值：

```python
transition_duration=0.5
```

差异较大时使用：

```python
transition_duration=0.8
# 或
transition_duration=1.0
```

### 限制单步关节变化

```python
max_joint_step = max_joint_velocity * dt
joint_delta = next_joint_pos - current_joint_pos
joint_delta = torch.clamp(joint_delta, -max_joint_step, max_joint_step)
next_joint_pos = current_joint_pos + joint_delta
```

只裁剪关节速度不够，最好同时限制单步位置变化。

### 计算关节速度

```python
joint_vel = (next_joint_pos - current_joint_pos) / dt
```

然后按软速度限制裁剪。`dt` 必须与更新频率一致：

```python
dt = sim.dt                 # 每个 physics step 更新
# 或
dt = sim.dt * decimation   # 每个 environment step 更新
```

## 8. Percent 观测与周期编码

直接把 `alpha` 作为 percent 观测，会出现正常行走和切换之间的边界跳变：

```text
正常行走 alpha = 1
开始切换 alpha = 0
```

这会形成 `1 -> 0` 的输入突变，影响 policy 学习。

### 不推荐只使用 sin

```python
percent = torch.sin(torch.pi * alpha)
```

它无法区分 `alpha=0.2` 和 `alpha=0.8`，因为两者可能得到相同数值。

### 推荐使用 sin/cos 周期编码

```python
phase = 2.0 * torch.pi * alpha
progress_sin = torch.sin(phase)
progress_cos = torch.cos(phase)
```

当 `alpha=0` 和 `alpha=1` 时：

```text
sin = 0
cos = 1
```

因此完成切换回到正常行走时，不会出现 `percent=1 -> percent=0` 的离散跳变。

示例接口：

```python
@property
def transition_phase(self) -> torch.Tensor:
    phase = 2.0 * torch.pi * self.transition_alpha
    return torch.stack(
        [torch.sin(phase), torch.cos(phase)],
        dim=-1,
    )
```

推荐额外提供连续的目标差：

```python
structure_delta = target_joint_pos - current_joint_pos
```

`structure_delta` 能够表达当前还需要移动多少，比单独的结构 ID 更适合 policy。

## 9. 观测设计

推荐 policy 和 critic 都观察：

```text
当前结构目标的连续表达
transition_phase_sin
transition_phase_cos
structure_delta
当前关节位置
当前关节速度
last_action
```

可以保留 `structure_id` 作为目标类别信息，但不要只依赖原始整数 `0/1/2/3`。更稳定的表达方式是：

- target joint position
- target joint position - current joint position
- one-hot structure ID
- sin/cos transition phase

如果使用历史观测：

```python
history_length = 5
```

历史观测能帮助 policy 感知短期运动变化，但不能替代显式的 phase 和目标差观测。`percent` 应由 command controller 的 transition state 产生，而不是让 policy 猜。

## 10. 奖励设计

### 切换跟踪奖励

切换期间跟踪插值参考位置：

```python
actual_joint_pos = asset.data.joint_pos[:, joint_ids]
reference_joint_pos = command_term.current_joint_pos
error = actual_joint_pos - reference_joint_pos
tracking_reward = torch.exp(
    -tracking_scale * torch.sum(torch.square(error), dim=1)
)
```

只在切换期间执行：

```python
tracking_reward *= transition_active.float()
```

这样 normal walking 阶段不会持续施加构型切换奖励。

### 最终目标奖励

切换完成时评价最终目标：

```python
final_error = actual_joint_pos - command_term.target_joint_pos
complete = (
    (~command_term.transition_active)
    & (torch.sum(torch.square(final_error), dim=1) < tolerance)
)
```

完成奖励只能在完成事件发生的那一帧给一次。需要保存上一帧的 active 状态：

```python
completed = was_transitioning & ~transition_active
```

### 平滑和稳定奖励

建议同时保留：

```text
关节加速度惩罚
关节速度惩罚
功率/能耗惩罚
身体姿态稳定奖励
非法接触惩罚
速度跟踪奖励
```

否则 policy 可能通过大力矩、跳跃、摔倒或冲击来完成构型切换。

## 11. 直接写仿真与真正 policy 学习的区别

当前控制器中的：

```python
asset.write_joint_state_to_sim(
    next_joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
)
```

会直接改变仿真状态。这适合：

- reset
- 初始化姿态
- 验证 command 和插值路径

但如果在训练过程中持续调用，policy 看到的是环境强制完成的切换，policy 并没有学会如何通过动作完成切换。

真正 policy 学习模式应该是：

```text
StructureCommand 生成 reference_joint_pos
    -> policy 输出 action
    -> JointPositionAction 或 torque controller 执行动作
    -> reward 评价 actual_joint_pos 与 reference_joint_pos 的误差
```

建议逐步迁移：

1. reset 阶段允许 `write_joint_state_to_sim`。
2. 运行中只更新 reference，不直接写 joint state。
3. 让 policy 的 joint position 或 torque action 跟踪 reference。
4. 用 tracking、completion、稳定性和能耗奖励训练。

## 12. 状态机

推荐四状态：

```text
IDLE
    当前构型稳定

REQUESTED
    收到新的结构请求

TRANSITIONING
    alpha 从 0 增加到 1

HOLD
    到达目标构型并保持
```

基本流程：

```python
if state == IDLE and switch_requested:
    start_transition()
elif state == TRANSITIONING:
    update_alpha()
    update_reference()
    if alpha >= 1.0:
        state = HOLD
elif state == HOLD:
    keep_target()
```

切换进行中收到新请求时，推荐暂时忽略或排队，不要直接反向重启插值。

## 13. 更新频率

`CommandTerm._update_command()` 是否每个 environment step 调用，取决于 Isaac Lab 的 CommandManager 生命周期。

如果每个 environment step 调用：

```python
dt = sim.dt * decimation
```

如果需要每个 physics step 更新：

```python
dt = sim.dt
```

不要只定义一个 `update()` 方法并假设框架会自动调用。必须确认调用入口：

- 环境 `step()`
- physics callback
- 高频 interval event
- command manager 的 update 生命周期

对平滑插值，优先选择每个 physics step 的明确更新入口；初始验证可以先使用每个 environment step。

## 14. 训练阶段

### 阶段一：单构型行走

```text
固定 structure_id=0
关闭运行时切换
先学会稳定行走
```

### 阶段二：四构型 reset 随机化

```text
episode reset 随机选择 0/1/2/3
episode 内保持不变
```

### 阶段三：环境触发运行中切换

```text
固定间隔触发 command 改变
使用 0.5-1.0 秒插值
训练跟踪参考轨迹和稳定性
```

### 阶段四：policy 主动切换

将切换请求加入 action：

```text
保持当前构型
切换到构型 0
切换到构型 1
切换到构型 2
切换到构型 3
```

最终由 policy 自己决定何时切换、切换到哪种结构以及是否等待当前 transition 完成。

## 15. 验证清单

### 构型解析

- 四个结构 ID 都能解析。
- 四个结构的位置 tensor 形状一致。
- 四个结构的值确实不同。
- 所有关节正则都能匹配至少一个 joint。

### Command

- command shape 为 `[num_envs, 1]`。
- reset 时为每个环境独立采样。
- 不同环境可以同时处于不同结构。
- command 变化只启动一次 transition。
- transition 未完成时不会反复重启。

### 插值

- alpha 始终位于 `[0, 1]`。
- alpha 到 1 后保持稳定。
- 位置和速度不超过软限制。
- 目标变化大时仍无单步跳变。
- reset 会清理对应环境的旧状态。

### 观测和奖励

- phase sin/cos 在切换边界连续。
- structure delta 与实际目标一致。
- tracking reward 只在切换期间有效。
- completion reward 只发放一次。
- reward 不会因为环境强制写入而虚高。

### 训练运行

- rough onepolicy 任务能创建。
- flat onepolicy 任务能创建。
- 短时间 rollout 中能观察到多个 structure ID。
- 切换期间机器人不立即失稳。
- policy 不依赖单独的离散 ID 跳变来判断 phase。

## 16. 当前实现的注意事项

当前 `StructureCommand` 已经具备：

- 构型字典加载
- 结构位置 tensor 生成
- reset 随机采样
- command 变化检测
- smoothstep 位置插值
- 关节位置和速度写入

仍建议后续补充：

1. sin/cos phase 观测接口。
2. structure delta 观测接口。
3. transition tracking reward 和 completion reward。
4. 最大关节速度/单步位置变化限制。
5. 运行中不直接写状态、改为 policy 跟踪参考轨迹。
6. 对 `CommandTerm` 的实际 update 生命周期做 Isaac Lab 运行时确认。

文档目标不是替代配置，而是作为 `StructureCommand` 的设计和训练实现参考。
