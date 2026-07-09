
## Initial Request

参看 /home/ymr/mingrui/research/project_retargeting/retargeting/temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md。现在我希望开始 Phase 2 的工作。

你可以参考之前完成的tasks（.agents/tasks）

## Phase 2 Plan

目标：在 Phase 1 已完成 `src/retargeting/` package 化的基础上，开始“抽配置”。Phase 2 不重写核心算法、不拆 ROS/hardware、不重排 assets/data；只把当前 offline replay / viser / retarget smoke 路径中最关键的 robot 与 retargeting 硬编码迁出到可校验配置，并保持旧调用方式兼容。

### 约束与边界

- 保持现有行为优先，Phase 0/Phase 1 的 headless tests 必须继续通过。
- 默认只覆盖 headless/offline 路径：`retargeting_replay.py`、`viser_retargeting_visualize.py`、`RobotPinocchio`、`RobotAdaptor`、`VectorWristJointOptimizer` smoke path。
- 不启动 RViz、Open3D GUI、MuJoCo viewer、camera、real robot 或 ROS launch。
- 不在本阶段移动 `assets/`、`data/` 或 ROS description package；配置可以继续引用当前 symlink-based URDF 路径。
- 不把 hardware IP/topic 迁入真实可用配置；如需要，只创建 `.example` 或后续占位计划，避免把实验室环境变成默认路径。
- 不引入大依赖；优先使用已安装/轻量方案。如果需要 YAML loader，先确认环境是否已有 `PyYAML`，否则评估使用 TOML 或极小配置 parser，避免为了配置抽取破坏安装。

### Phase 2 范围

第一批配置只覆盖当前实际可验证链路：

1. Robot config：
   - Panda + Leap Paxini：offline replay 默认 `leap`
   - Panda + Shadow：offline replay 可选 `shadow`
   - 可选补充 Panda + Leap Tac3D：保留给 MuJoCo/teleop 现有路径，但不作为首要验证目标

2. Retargeting config：
   - 当前默认 `VECTOR_WRIST_JOINT`
   - `setting_id = 3`
   - `ablation_option = 0`
   - Leap / Shadow 的 link pairs
   - `wrist_link_name`
   - optimizer params：`huber_delta = 0.02`、`opt_ftol_abs = 1e-5`、`opt_maxtime = 0.05`
   - joint limit override：当前 `retarget_optimizer.py` 中 `[9, 10, 13, 14, 17, 18] -> lower=-0.2`

3. App / replay config：
   - default replay data path
   - hand type / robot config path
   - retarget config path
   - start/end/stride/fps/port/no_robot_mesh/trail_length

### Step 1：建立配置目录和配置文件

新建最小目录结构：

```text
configs/
├── robots/
│   ├── panda_leap_paxini.yaml
│   ├── panda_leap_tac3d.yaml
│   └── panda_shadow.yaml
├── retargeting/
│   └── vector_wrist_joint.yaml
└── apps/
    └── replay_avp.yaml
```

配置内容先保持直接、显式、易读，不追求最终完美 schema。

Robot config 字段计划：

```yaml
name: panda_leap_paxini
hand_type: leap
model:
  type: urdf
  path: assets/panda_leap_paxini.urdf
  path_is_symlink: true
actuated_joints:
  - panda_joint1
  - ...
touch_joints: []
initial_qpos:
  - ...
visual_frame_names:
  - wrist
  - thumb_tip_center
  - ...
wrist_frame_name: wrist
human_hand_scale: 1.5
```

Retargeting config 字段计划：

```yaml
type: VECTOR_WRIST_JOINT
setting_id: 3
ablation_option: 0
optimizer:
  class: VectorWristJointOptimizer
  params:
    huber_delta: 0.02
    opt_ftol_abs: 0.00001
    opt_maxtime: 0.05
targets:
  leap:
    wrist_link_name: wrist
    link_pairs:
      - [world, thumb_tip_center]
      - ...
  shadow:
    wrist_link_name: ee_link
    link_pairs:
      - [world, thtip]
      - ...
joint_limit_overrides:
  - indices: [9, 10, 13, 14, 17, 18]
    lower: -0.2
```

App config 字段计划：

```yaml
data: tests/fixtures/avp_short_replay.npz
robot: configs/robots/panda_leap_paxini.yaml
retargeting: configs/retargeting/vector_wrist_joint.yaml
start: 0
end: -1
stride: 1
viewer:
  fps: 30.0
  port: 8080
  no_robot_mesh: false
  trail_length: 120
```

### Step 2：新增配置加载与校验模块

新增 `src/retargeting/config/`：

```text
src/retargeting/config/
├── __init__.py
├── io.py
├── schema.py
└── defaults.py
```

职责：

- `io.py`：加载配置文件，解析相对路径为 repo/root-relative 或 config-file-relative 路径。
- `schema.py`：定义 dataclass，例如 `RobotConfig`、`RetargetingConfig`、`ReplayAppConfig`。
- `defaults.py`：提供 `default_robot_config_for_hand_type("leap"|"shadow")`，保持旧 API 能从 `hand_type` 自动找到配置。

最小校验：

- robot model path 存在；如是 symlink，最终 target 存在。
- actuated joint 数量与 initial qpos 长度一致。
- `hand_type` 在允许值内。
- `wrist_frame_name` 出现在 visual frames 或 robot frame names 中。
- retarget link pairs 非空，且每项长度为 2。
- optimizer params 包含当前必要字段。
- joint limit override indices 是整数列表。

### Step 3：把 replay context 改为从 RobotConfig 构造

目标文件：`src/retargeting/retargeting_replay.py`

计划改动：

1. 保留现有 `create_robot_replay_context(hand_type="leap")` 兼容入口。
2. 新增 `create_robot_replay_context_from_config(robot_config)`。
3. 将以下硬编码迁入配置：
   - `LEAP_ACTUATED_JOINTS_NAME`
   - `SHADOW_ACTUATED_JOINTS_NAME`
   - `LEAP_VISUAL_FRAME_NAMES`
   - `SHADOW_VISUAL_FRAME_NAMES`
   - `get_default_init_joint_pos()`
   - `assets/panda_leap_paxini.urdf`
   - `assets/panda_shadow.urdf`
   - `wrist` / `ee_link`
   - `human_hand_scale`
4. `build_retarget_replay_frames()` 增加可选参数：
   - `robot_config: RobotConfig | None = None`
   - `retargeting_config: RetargetingConfig | None = None`
   - 或 path 版本：`robot_config_path`、`retargeting_config_path`
5. 默认行为保持不变：不传配置时，`hand_type="leap"` 仍等价于当前 Panda+Leap Paxini。

### Step 4：让 retargeter / optimizer 使用 RetargetingConfig

目标文件：

- `src/retargeting/robot_teleoperation.py`
- `src/retargeting/retarget_optimizer.py`
- `src/retargeting/retargeting_replay.py`

计划改动：

1. `RobotTeleoperation.__init__()` 增加可选 `retargeting_config`。
2. 若传入 config，则从 config 构造：
   - retarget type
   - ablation option
   - target link pairs
   - wrist link name
   - optimizer params
   - human hand scale
3. 若不传 config，保留现有硬编码默认路径，避免一次性 break。
4. `RetargetOptimizer.__init__()` 增加可选 `joint_limit_overrides`，把当前写死的 joint limit override 从 config 注入。
5. 对当前 tests 的 `VectorWristJointOptimizer(...)` 直接构造方式保持兼容；不强迫所有调用者立刻传 config。

### Step 5：更新 viser/offline replay CLI 支持 config

目标文件：

- `src/retargeting/viser_retargeting_visualize.py`
- `pyproject.toml` 中现有 `retargeting-replay`

计划改动：

1. `viser_retargeting_visualize.py` 增加参数：
   - `--config configs/apps/replay_avp.yaml`
   - `--robot configs/robots/panda_leap_paxini.yaml`
   - `--retarget configs/retargeting/vector_wrist_joint.yaml`
2. 命令行显式参数优先级高于 app config，例如 `--data`、`--end 100` 可以覆盖 config 中的值。
3. 保留当前命令完全可用：
   ```bash
   python -m retargeting.viser_retargeting_visualize \
     --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
     --hand-type leap \
     --end 100
   ```
4. 增加 config-first 启动方式：
   ```bash
   python -m retargeting.viser_retargeting_visualize \
     --config configs/apps/replay_avp.yaml
   ```

### Step 6：测试覆盖

新增或更新 tests：

1. `tests/test_config_loading.py`
   - 能加载 `configs/robots/panda_leap_paxini.yaml`
   - 能加载 `configs/robots/panda_shadow.yaml`
   - 能加载 `configs/retargeting/vector_wrist_joint.yaml`
   - 路径解析后 URDF target 存在
   - joint 数量与 initial qpos 长度一致

2. 更新 `tests/test_replay_smoke.py`
   - RobotAdaptor round-trip 改用 robot config 构造一次
   - optimizer smoke 使用 retargeting config 的 link pairs 和 params 构造一次
   - 保留至少一个旧构造方式测试，确认兼容

3. 更新 `tests/test_visualization_replay.py`
   - `build_retarget_replay_frames(..., robot_config_path=..., retargeting_config_path=...)` 能重新 retarget 并构建 frame

4. CLI smoke：
   - 只跑 `--help` 或 parser/config merge，不启动 server 长循环。

验证命令：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

如修改 packaging 或新增 package data，需要额外验证：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

### Step 7：完成标准

- `configs/robots/`、`configs/retargeting/`、`configs/apps/` 存在，并至少覆盖 Panda+Leap Paxini、Panda+Shadow、VectorWristJoint replay。
- `src/retargeting/config/` 提供 typed config loading 和最小校验。
- offline replay / viser 路径可以通过 config 构造 robot context 和 retargeter。
- 旧 API 仍可用：
  - `build_retarget_replay_frames(data_file=..., hand_type="leap", ...)`
  - `python -m retargeting.viser_retargeting_visualize --data ... --hand-type leap`
- 新 config-first API 可用：
  - `python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml`
- 当前 headless tests 通过。
- 不启动 GUI/ROS/hardware。

### 风险与处理

- YAML dependency 风险：如果 `PyYAML` 不在环境中，先确认是否可添加依赖；若不合适，改用 TOML 或最小 JSON/TOML 配置，避免 Phase 2 被依赖安装阻塞。
- 路径解析风险：当前 URDF 依赖 symlink 和 cwd。Phase 2 只把路径显式写入配置并校验 target，不在本阶段移 assets。
- 行为漂移风险：先让默认 config 复刻当前硬编码值，再修改调用链；每一步跑 smoke tests。
- API 破坏风险：所有新增参数使用 optional default，不删除现有 `hand_type`、`data_file` 等入口。
- 配置膨胀风险：只抽 headless/offline 可验证参数；hardware topics/IP 放到 `.example` 或 Phase 3/4，不把不可测配置混进核心路径。

## Phase 2 Implementation Record

状态：已完成。

### 已完成的主要改动

1. 新增配置目录和标准 YAML 配置文件：
   - `configs/robots/panda_leap_paxini.yaml`
   - `configs/robots/panda_leap_tac3d.yaml`
   - `configs/robots/panda_shadow.yaml`
   - `configs/retargeting/vector_wrist_joint.yaml`
   - `configs/apps/replay_avp.yaml`

2. Robot config 已覆盖：
   - `hand_type`
   - URDF model type/path/symlink 标记
   - actuated joints
   - touch joints
   - initial qpos
   - visual frame names
   - wrist frame name
   - human hand scale

3. Retargeting config 已覆盖：
   - `VECTOR_WRIST_JOINT`
   - `setting_id = 3`
   - `ablation_option = 0`
   - Leap / Shadow link pairs
   - wrist link name
   - optimizer params
   - joint limit override `[9, 10, 13, 14, 17, 18] -> lower=-0.2`

4. App config 已覆盖 replay viewer 默认参数：
   - data path
   - robot config path
   - retargeting config path
   - start/end/stride
   - viewer fps/port/no robot mesh/trail length

5. 新增 `src/retargeting/config/`：
   - `__init__.py`
   - `io.py`
   - `schema.py`
   - `defaults.py`

   其中 `schema.py` 提供：
   - `RobotModelConfig`
   - `RobotConfig`
   - `RetargetTargetConfig`
   - `JointLimitOverride`
   - `RetargetingConfig`
   - `ViewerConfig`
   - `ReplayAppConfig`

6. 安装并记录 YAML 依赖：
   - 已安装 `PyYAML 6.0.3` 到 `/home/ymr/miniconda3/envs/retargeting`
   - `pyproject.toml` 增加：
     ```toml
     dependencies = [
         "PyYAML>=6.0",
     ]
     ```
   - `src/retargeting/config/io.py` 使用 `yaml.safe_load`

7. `src/retargeting/retargeting_replay.py` 已支持配置入口：
   - `create_robot_replay_context_from_config(robot_config)`
   - `build_retarget_replay_frames(..., robot_config=..., retargeting_config=...)`
   - `build_retarget_replay_frames(..., robot_config_path=..., retargeting_config_path=...)`

   同时保留旧入口：
   - `create_robot_replay_context(hand_type="leap")`
   - `build_retarget_replay_frames(data_file=..., hand_type="leap", ...)`

8. `src/retargeting/robot_teleoperation.py` 已支持 `retargeting_config`：
   - 从 config 读取 retarget type
   - 从 config 读取 ablation option
   - 从 config 读取 target link pairs
   - 从 config 读取 wrist link name
   - 从 config 读取 optimizer params
   - 从 config 读取 joint limit overrides
   - 支持传入 `human_hand_scale`

9. `src/retargeting/retarget_optimizer.py` 已支持可选 `joint_limit_overrides`：
   - 默认仍复刻原行为
   - 显式配置时从 `RetargetingConfig` 注入

10. `src/retargeting/viser_retargeting_visualize.py` 已支持配置驱动：
    - `--config`
    - `--robot`
    - `--retarget`
    - CLI 显式参数覆盖 app config
    - 原 `--data` / `--hand-type` / `--end` 方式仍可用

11. 按后续要求更新 viewer 默认状态：
    - `Show trails` 默认关闭
    - GUI 中仍可手动打开 trails

### 当前推荐启动命令

Leap Paxini，offline trajectory，重新计算 retarget qpos，`end=1000`：

```bash
cd /home/ymr/mingrui/research/project_retargeting/retargeting

/home/ymr/miniconda3/envs/retargeting/bin/python \
  -m retargeting.viser_retargeting_visualize \
  --robot configs/robots/panda_leap_paxini.yaml \
  --retarget configs/retargeting/vector_wrist_joint.yaml \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --end 1000 \
  --port 8080
```

也可以使用 app config：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python \
  -m retargeting.viser_retargeting_visualize \
  --config configs/apps/replay_avp.yaml
```

### 测试与验证记录

1. PyYAML 安装验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import yaml; print(yaml.__version__)"
   ```
   结果：`6.0.3`

2. 配置加载验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "from retargeting.config import load_robot_config, load_retargeting_config, load_replay_app_config; print(load_robot_config('configs/robots/panda_leap_paxini.yaml').name); print(load_retargeting_config('configs/retargeting/vector_wrist_joint.yaml').targets_for('leap').wrist_link_name); print(load_replay_app_config('configs/apps/replay_avp.yaml').viewer.port)"
   ```
   结果：
   ```text
   panda_leap_paxini
   wrist
   8080
   ```

3. Headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
   结果：`12 passed`

4. CLI help：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
   ```
   结果：可见 `--config`、`--robot`、`--retarget` 等参数。

### 未纳入 Phase 2 的内容

- 未移动 `assets/`、`data/` 或 ROS description package。
- 未拆分 core / ROS / hardware。
- 未配置真实 hardware IP/topic。
- 未重写 optimizer API。
- 未启动 RViz、Open3D GUI、MuJoCo viewer、camera、real robot 或 ROS launch。
