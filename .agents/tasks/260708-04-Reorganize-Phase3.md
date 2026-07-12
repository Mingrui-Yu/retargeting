
## Initial Request

参看 /home/ymr/mingrui/research/project_retargeting/retargeting/temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md。现在我希望开始 Phase 3 的工作。

你可以参考之前完成的tasks（.agents/tasks/）

## Phase 3 Plan

目标：在 Phase 1 已完成 package 化、Phase 2 已完成第一批配置化的基础上，拆清 `retargeting` core、ROS integration、hardware/runtime adapters 的依赖边界。Phase 3 的核心完成标准是：`retargeting` 作为纯 Python core/offline package，在没有 ROS、camera、real hardware、GUI viewer 的环境中仍可 import 和运行 headless replay tests；ROS、RViz、real robot、高频控制等入口进入单独的 adapter/package 边界，并通过薄 wrapper 继续兼容旧路径。

### 约束与边界

- 保持当前行为优先，Phase 0/1/2 的 headless tests 必须继续通过。
- 不启动 RViz、Open3D GUI、MuJoCo viewer、camera、Realsense、real robot 或 ROS launch。
- 不在本阶段重写 optimizer API，不改变 retargeting objective 数值逻辑。
- 不在本阶段移动 `assets/`、`data/`、ROS description package 或大实验输出；这些属于后续 Phase 4。
- 不删除旧 `ws_ros2/src/retargeting_benchmark/src/*.py` 入口；迁出的文件先保留 wrapper 或原路径 adapter，避免 ROS install(PROGRAMS)、用户脚本和 launch 文件一次性失效。
- `src/retargeting/` 是 core/offline source of truth；新的 ROS/hardware 代码不能让 core 反向 import ROS package。
- 任何需要可选依赖的模块都必须 lazy import 或被移出 core 默认 import 链，缺依赖时给出明确错误或测试 skip。

### Phase 3 范围

第一批只处理能通过 headless 验证的边界拆分：

1. Core package 边界：
   - `src/retargeting/__init__.py` import 必须不触发 `rclpy`、ROS messages、`cv_bridge`、`pynput`、Realsense、real hardware、MuJoCo viewer、Open3D GUI 或 `viser`。
   - offline replay、config loading、Pinocchio model、RobotAdaptor、retarget optimizer 继续留在 core。
   - `robot_teleoperation.py` 中可复用的纯 retargeting session 逻辑留在 core，但 ROS/hardware/viewer side effects 必须通过 adapter 或 lazy import 隔离。

2. ROS package 边界：
   - 新增 `src/retargeting_ros/`，作为 Python-level ROS adapter package。
   - ROS node、RViz publisher、ROS message conversion、real robot ROS communication、高频控制 node 都归到 `retargeting_ros` 或保留在旧 ROS package 下作为 wrapper。
   - `retargeting_ros` 可以 import `retargeting`，但 `retargeting` 不能 import `retargeting_ros`。

3. Runtime adapter 边界：
   - hardware backend、MuJoCo visualization backend、viser visualization backend、Vision Pro live streamer 都通过 protocol/lazy import 接入。
   - offline replay 默认路径不依赖 live Vision Pro、ROS、hardware 或 viewer。
   - `VisionProDetector` 保持当前 lazy `avp_stream` 策略，必要时抽出 input protocol，避免 live streamer 污染 replay。

4. 旧路径兼容：
   - 已迁入 `src/retargeting/` 的旧路径文件继续使用 `_retargeting_compat.py` wrapper。
   - 被迁入 `src/retargeting_ros/` 的 ROS 文件也保留旧路径 wrapper。
   - 暂未迁移的 ROS launch/CMake entrypoint 文件名不变。

### Step 1：依赖审计与 import contract

先用 `rg` 建立当前依赖地图，记录哪些模块在 import 阶段触发可选依赖：

- ROS：`rclpy`、`sensor_msgs`、`std_msgs`、`geometry_msgs`、`visualization_msgs`、`tf2_ros`、`cv_bridge`
- hardware/live input：`robot_real`、`RobotReal`、`avp_stream`、Realsense/camera、`pynput`
- GUI/simulation：`mujoco.viewer`、Open3D GUI、`viser`

新增或更新 headless import tests：

```python
def test_core_import_has_no_ros_dependency():
    import retargeting


def test_core_offline_modules_import_without_ros():
    from retargeting import offline_replay, retargeting_replay
    from retargeting.robot_pinocchio import RobotPinocchio
    from retargeting.robot_adaptor import RobotAdaptor
```

如果环境没有 ROS，测试应证明 core import 不需要 ROS；如果环境有 ROS，也要通过 monkeypatch/import hook 防止误把 ROS 可用性当成通过条件。

### Step 2：建立 adapter/protocol 抽象

新增轻量 protocol 模块，优先只描述当前代码已经需要的最小接口，不做过度抽象：

```text
src/retargeting/
├── backends/
│   ├── __init__.py
│   └── base.py
├── inputs/
│   ├── __init__.py
│   └── base.py
└── visualization/
    ├── __init__.py
    └── base.py
```

计划接口：

```python
class RobotBackend(Protocol):
    def get_joint_pos(self) -> np.ndarray: ...
    def command_joint_pos(self, qpos: np.ndarray) -> None: ...
    def step(self) -> None: ...


class HandInput(Protocol):
    def get_observation(self) -> HandObservation: ...


class ReplayVisualizer(Protocol):
    def update_frame(self, frame: object) -> None: ...
```

实现原则：

- Protocol 只依赖 stdlib typing 和 numpy。
- 不在 protocol 模块里 import ROS、MuJoCo、viser、Open3D 或 hardware driver。
- 先让 `robot_teleoperation.py` / replay runner 能接收 optional backend/input，而不是直接实例化 ROS/hardware 类。

### Step 3：拆出 `retargeting_ros`

新增：

```text
src/retargeting_ros/
├── __init__.py
├── messages.py
├── rviz.py
├── real_robot.py
├── nodes/
│   ├── __init__.py
│   ├── teleoperation.py
│   ├── robot_control.py
│   ├── robot_real_high_freq.py
│   └── virtual_robot.py
└── launch_helpers/
    └── __init__.py
```

迁移候选：

- `ws_ros2/src/retargeting_benchmark/src/utils/utils_ros.py` -> `src/retargeting_ros/messages.py`
- `ws_ros2/src/retargeting_benchmark/src/rviz_visualize.py` -> `src/retargeting_ros/rviz.py`
- `ws_ros2/src/retargeting_benchmark/src/robot_real.py` -> `src/retargeting_ros/real_robot.py`
- `main_robot_teleoperation.py`、`main_robot_control.py`、`main_robot_real_high_freq.py`、`main_virtual_robot.py` 中的 node class -> `src/retargeting_ros/nodes/`

兼容策略：

- 旧路径文件替换为 wrapper：
  ```python
  from _retargeting_compat import ensure_retargeting_src_on_path

  ensure_retargeting_src_on_path()

  from retargeting_ros.<module> import *  # noqa: F401,F403
  ```
- 有 `main()` 的旧脚本继续调用迁入模块的 `main()`。
- CMake/launch 中引用的脚本文件名保持不变。

### Step 4：隔离 hardware 与 live runtime side effects

处理 `RobotTeleoperation` 和相关调用链：

1. 保留 offline replay 构造路径：
   - `mujoco_vis=False`
   - `use_real_hardware=False`
   - 不 import `RobotReal`
   - 不 import ROS node
2. 将 real hardware 控制替换为 `RobotBackend` optional dependency：
   - core 中只保存 `backend: RobotBackend | None`
   - ROS real robot backend 在 `retargeting_ros.real_robot` 中实现
3. 将 MuJoCo viewer import 限制在明确开启 `mujoco_vis=True` 的路径，并保持已有 lazy import。
4. 将 keyboard/`pynput` 依赖从 core 默认 import 链移出；只在交互控制脚本或 ROS/hardware adapter 中 import。
5. 对缺少 optional dependency 的路径给出明确错误，例如：
   - `"ROS support requires rclpy and should be run from the ROS environment."`
   - `"MuJoCo visualization requires mujoco and a display-capable environment."`

### Step 5：整理 utils 依赖边界

当前 `src/retargeting/utils/` 中仍有 ROS、Open3D、MuJoCo、keyboard helper。Phase 3 先做最小拆分：

- ROS helper 迁到 `retargeting_ros.messages`。
- keyboard helper 只由 interactive/hardware scripts import，core 不直接 import。
- Open3D helper 保留在 `retargeting.utils` 但不被 core 默认 import；如需要，可后续迁到 `retargeting.visualization.open3d_utils`。
- MuJoCo conversion helper 保留为 optional asset tooling，不被 offline replay/import tests 覆盖；后续 Phase 4 处理 assets 时再进一步整理。

新增依赖边界测试：

```bash
rg -n "rclpy|sensor_msgs|std_msgs|geometry_msgs|visualization_msgs|cv_bridge|tf2_ros" src/retargeting
```

允许的临时例外需要写清楚原因，例如迁移过渡期未拆出的 wrapper，但目标是 core 下不再出现 ROS import。

### Step 6：更新 packaging

更新 `pyproject.toml`：

- package discovery 包含：
  ```toml
  include = ["retargeting*", "retargeting_ros*"]
  ```
- 不把 ROS 依赖加入 core `dependencies`。
- 如需要，增加 optional extras：
  ```toml
  [project.optional-dependencies]
  ros = []
  vis = []
  mujoco = []
  ```
  具体依赖可先留空或只记录说明，避免在非 ROS 环境中误装不可用依赖。

Console scripts 策略：

- 保留已有 headless/offline `retargeting-replay`。
- ROS/hardware console scripts 暂不作为默认入口暴露，除非它们 import 阶段不需要 ROS 或仅在 ROS 环境中运行。
- 如果新增 ROS scripts，应指向 `retargeting_ros.nodes.*:main`，并在文档/错误信息中明确需要 ROS 环境。

### Step 7：测试与验证

默认验证命令：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

新增 import contract 验证：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting"
/home/ymr/miniconda3/envs/retargeting/bin/python -c "from retargeting.robot_pinocchio import RobotPinocchio; from retargeting.robot_adaptor import RobotAdaptor"
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting_ros"
```

配置/回放验证继续覆盖 Phase 2：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --help
/home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
```

旧路径 wrapper 验证：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import sys; sys.path.insert(0, 'ws_ros2/src/retargeting_benchmark/src'); import robot_pinocchio; import offline_replay"
```

ROS/hardware 相关验证只做 import-wrapper 或静态检查；不启动 ROS launch、RViz、camera 或 real robot。

### Step 8：完成标准

- `python -c "import retargeting"` 在无 ROS 环境中成功。
- `python -c "import retargeting_ros"` 成功，且不在 package import 阶段启动 node 或连接 hardware。
- `src/retargeting/` 中不再有直接 ROS import；如有临时例外，必须明确记录并附后续处理项。
- offline replay、config loading、Pinocchio、RobotAdaptor、optimizer smoke tests 全部通过。
- ROS/RViz/hardware 代码已归入 `retargeting_ros` 或被旧路径 wrapper 隔离。
- 旧路径脚本仍可 import，ROS package 中已安装的脚本文件名不变。
- `pyproject.toml` package discovery 覆盖 `retargeting` 和 `retargeting_ros`，core dependencies 不新增 ROS/hardware/GUI 强依赖。

### 风险与处理

- ROS 文件迁移风险：先迁 Python node/helper，保持 launch/CMake 文件名和旧脚本路径不变；每个旧脚本用 wrapper 单独验证 import。
- 循环 import 风险：`retargeting_ros` 可以依赖 `retargeting`，反向依赖一律禁止；必要时把共享 dataclass/protocol 放回 `retargeting.backends` 或 `retargeting.inputs`。
- 可选依赖缺失风险：把 import 移到函数内部，缺依赖时用清晰 `ImportError`；测试中使用 `pytest.importorskip` 或 import hook 验证 core 不依赖它们。
- 行为漂移风险：不改 retargeting objective 和配置默认值；每次迁移后跑 `tests`，特别是 replay smoke 和 config loading。
- 迁移范围膨胀风险：Phase 3 只处理 core/ROS/hardware 边界；assets/data 清理、optimizer typed API、README 重写留给后续阶段。

## Phase 3 Implementation Record

状态：已完成。

### 已完成的主要改动

1. 新增 core-side protocol 包，作为后续 runtime adapter 的稳定边界：
   - `src/retargeting/backends/__init__.py`
   - `src/retargeting/backends/base.py`
   - `src/retargeting/inputs/__init__.py`
   - `src/retargeting/inputs/base.py`
   - `src/retargeting/visualization/__init__.py`
   - `src/retargeting/visualization/base.py`

   目前包含最小接口：
   - `RobotBackend`
   - `HandObservation`
   - `HandInput`
   - `ReplayVisualizer`

2. 新增 Python-level ROS adapter package：
   - `src/retargeting_ros/__init__.py`
   - `src/retargeting_ros/_optional.py`
   - `src/retargeting_ros/messages.py`
   - `src/retargeting_ros/rviz.py`
   - `src/retargeting_ros/real_robot.py`
   - `src/retargeting_ros/nodes/__init__.py`
   - `src/retargeting_ros/nodes/robot_real_high_freq.py`
   - `src/retargeting_ros/nodes/virtual_robot.py`
   - `src/retargeting_ros/launch_helpers/__init__.py`

3. 将 ROS message / RViz / real robot 相关实现从 core 或旧 ROS script 路径迁到 `retargeting_ros`：
   - `utils_ros.py` 的 ROS message helper -> `retargeting_ros.messages`
   - `rviz_visualize.py` 的 `RvizVisualizer` -> `retargeting_ros.rviz`
   - `robot_real.py` 的 `RobotReal` -> `retargeting_ros.real_robot`
   - `main_robot_real_high_freq.py` 的 `RobotRealHighFreq` -> `retargeting_ros.nodes.robot_real_high_freq`
   - `main_virtual_robot.py` 的 `VirtualRobot` -> `retargeting_ros.nodes.virtual_robot`

4. 保留旧路径兼容 wrapper：
   - `ws_ros2/src/retargeting_benchmark/src/rviz_visualize.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_real.py`
   - `ws_ros2/src/retargeting_benchmark/src/main_robot_real_high_freq.py`
   - `ws_ros2/src/retargeting_benchmark/src/main_virtual_robot.py`
   - `ws_ros2/src/retargeting_benchmark/src/utils/utils_ros.py`

   其中 `main_robot_real_high_freq.py` 和 `main_virtual_robot.py` 的 executable bit 已恢复，避免直接执行或 ROS install 行为变化。

5. `src/retargeting/utils/utils_ros.py` 已改为兼容 alias：
   ```python
   from retargeting_ros.messages import *  # noqa: F401,F403
   ```

   这样 core 下不再直接 import `rclpy`、ROS messages、`cv_bridge`、`tf2_ros` 等 ROS package。

6. `retargeting_ros` 子模块已做 optional/lazy ROS dependency 处理：
   - `import retargeting_ros` 不初始化 ROS。
   - `import retargeting_ros.rviz`、`import retargeting_ros.real_robot`、`import retargeting_ros.nodes.*` 在无 ROS 环境中也可以成功。
   - 真正实例化 `RvizVisualizer`、`RobotReal`、`RobotRealHighFreq`、`VirtualRobot` 或执行 node `main()` 时才要求 ROS。
   - 缺少 ROS package 或 Leap service package 时，通过 `retargeting_ros._optional.missing_ros_error()` 给出明确错误。

7. 更新 `pyproject.toml`：
   - package discovery 覆盖：
     ```toml
     include = ["retargeting*", "retargeting_ros*"]
     ```
   - 保留已有 console script：
     ```toml
     retargeting-replay = "retargeting.viser_retargeting_visualize:main"
     ```
   - 增加 optional extras 占位：
     ```toml
     [project.optional-dependencies]
     ros = []
     vis = []
     mujoco = []
     ```

8. 新增 Phase 3 import boundary tests：
   - `tests/test_phase3_import_boundaries.py`

   覆盖：
   - core import 时阻断 ROS packages，确认 `retargeting`、`offline_replay`、`retargeting_replay`、`RobotPinocchio`、`RobotAdaptor` 不依赖 ROS。
   - `import retargeting_ros` 没有 runtime side effect。
   - 静态扫描 `src/retargeting`，确认不含直接 ROS import token。

### 验证记录

1. Core / adapter import 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; import retargeting_ros; from retargeting.robot_pinocchio import RobotPinocchio; from retargeting.robot_adaptor import RobotAdaptor"
   ```
   结果：通过。

2. Phase 3 boundary tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase3_import_boundaries.py
   ```
   结果：`3 passed`。

3. 完整 headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
   结果：`15 passed`。

4. Editable install：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .
   ```
   结果：通过。

   备注：第一次在 sandbox 内运行时因 DNS 无法解析 package index 失败；按环境规则使用提权网络重试后成功。

5. 旧路径 wrapper import 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import sys; sys.path.insert(0, 'ws_ros2/src/retargeting_benchmark/src'); import robot_pinocchio; import offline_replay; import rviz_visualize; import robot_real; import main_robot_real_high_freq; import main_virtual_robot; print(robot_pinocchio.RobotPinocchio); print(hasattr(rviz_visualize, 'RvizVisualizer')); print(hasattr(robot_real, 'RobotReal')); print(hasattr(main_robot_real_high_freq, 'RobotRealHighFreq')); print(hasattr(main_virtual_robot, 'VirtualRobot'))"
   ```
   结果：通过。

6. `retargeting_ros` 子模块无 ROS 环境 import 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting_ros; import retargeting_ros.rviz; import retargeting_ros.real_robot; import retargeting_ros.nodes.robot_real_high_freq; import retargeting_ros.nodes.virtual_robot; print('ok')"
   ```
   结果：`ok`。

7. CLI help 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --help
   /home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
   ```
   结果：通过。

8. Core ROS import 静态检查：
   ```bash
   rg -n "^import rclpy|^from rclpy|sensor_msgs|std_msgs|geometry_msgs|visualization_msgs|cv_bridge|tf2_ros|builtin_interfaces" src/retargeting
   ```
   结果：无输出。

### 未纳入 Phase 3 的内容

- 未启动 ROS launch、RViz、camera、Realsense、real robot、Open3D GUI 或 MuJoCo viewer。
- 未重写 optimizer API。
- 未移动 `assets/`、`data/` 或 ROS description package。
- 未整理大实验输出或 symlink-based asset layout。
- 未把 `robot_control.py`、`main_robot_control.py`、`main_robot_teleoperation.py` 完整迁入 `retargeting_ros`；本阶段先完成可验证的 ROS helper/RViz/real robot/high-frequency/virtual robot 边界拆分，剩余更高耦合 runtime 入口可在后续 Phase 继续迁移。
