
## Initial Request

参看 /home/ymr/mingrui/research/project_retargeting/retargeting/temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md。现在我希望开始 Phase 1 的工作。

Phase 1：让当前代码成为真正 Python package
- 新建 `src/retargeting/`。
- 先搬核心模块，不急着重写逻辑。
- 裸 import 改成 absolute package import。
- 在 `pyproject.toml` 加 console scripts。
- 旧路径脚本保留 wrapper，避免一次性 break。

## Phase 1 Plan

目标：在不重写 retargeting 逻辑、不改变 ROS/硬件运行语义的前提下，把当前核心 Python 代码迁移为可安装的 root-level `retargeting` package，并让旧入口继续可用。

### 约束与边界

- 保持 Phase 0 行为优先：现有 headless smoke tests 必须继续通过，尤其是 asset symlink、AVP replay fixture、Pinocchio model loading、RobotAdaptor mapping、optimizer single-frame smoke test。
- Phase 1 只做 packaging/import/entrypoint 层面的整理；不在本阶段配置化机器人参数、不重写 optimizer API、不搬 ROS description/assets/data。
- 核心 package 不主动引入新的 ROS 依赖；现有 ROS、RViz、hardware 相关文件可以先随代码迁入 package，但 console scripts 和测试默认只覆盖 headless/offline 路径。
- 旧路径 `ws_ros2/src/retargeting_benchmark/src/*.py` 保留为兼容 wrapper，避免 README、ROS install(PROGRAMS)、用户脚本一次性失效。

### Step 1：建立 package 骨架

1. 新建 `src/retargeting/` 和必要的 `__init__.py`。
2. 迁入第一批核心模块，优先选择当前测试和离线 replay 依赖的文件：
   - `retarget_optimizer.py`
   - `robot_adaptor.py`
   - `robot_pinocchio.py`
   - `offline_replay.py`
   - `retargeting_replay.py`
   - `vision_pro_detector.py`
   - `utils/`
3. 暂缓迁移高耦合 runtime 文件，除非 import 链要求必须迁入：
   - `robot_teleoperation.py`
   - `robot_control.py`
   - `robot_real.py`
   - `robot_mujoco.py`
   - `rviz_visualize.py`
   - plot/record/video/ROS launch helper 脚本
4. 保留原始文件名和类名，避免本阶段引入大范围 API rename。

### Step 2：改为 absolute package import

1. 将迁入模块内部的裸 import 改成 `retargeting.*` absolute import，例如：
   - `from robot_adaptor import RobotAdaptor` -> `from retargeting.robot_adaptor import RobotAdaptor`
   - `from utils.utils_calc import ...` -> `from retargeting.utils.utils_calc import ...`
   - `from offline_replay import ...` -> `from retargeting.offline_replay import ...`
2. 对测试中直接导入旧模块名的地方同步改为 package import。
3. 对暂未迁入但仍需被旧入口调用的模块，不让新 package 反向依赖旧 `ws_ros2/.../src` 路径。
4. 使用 `rg` 检查迁入代码里是否仍有核心裸 import 残留。

### Step 3：配置 `pyproject.toml`

1. 在现有 black/ruff 配置基础上追加 build-system 和 project metadata，采用 `setuptools` + `src` layout。
2. 增加 package discovery：
   - `where = ["src"]`
   - include `retargeting*`
3. 先声明最小 console scripts，覆盖低风险、headless/offline 入口：
   - `retargeting-replay = retargeting.viser_retargeting_visualize:main`（如本阶段迁移该文件）
   - 或先提供 `retargeting-offline-replay` / `retargeting-visualize-replay` 等薄 CLI，指向已有 `main()`。
4. 对硬件/ROS/RViz 入口暂不作为默认 console scripts 暴露，除非只是兼容 wrapper 且不会在 import 阶段强制启动 ROS/hardware。

### Step 4：旧路径 wrapper 兼容

1. 对已迁入 `src/retargeting/` 的旧路径文件，替换为薄 wrapper：
   ```python
   from retargeting.<module> import *

   if __name__ == "__main__":
       from retargeting.<module> import main

       main()
   ```
2. 对没有 `main()` 的库模块，只 re-export 公共符号，保证旧代码的 `from robot_pinocchio import RobotPinocchio` 仍可工作。
3. 保持 `ws_ros2/src/retargeting_benchmark/CMakeLists.txt` 中已 install 的 ROS scripts 文件名不变。
4. 如果某个旧脚本依赖当前工作目录解析 `assets/` 或 `data/`，本阶段不改变路径解析逻辑，只确保 wrapper 调用后行为一致。

### Step 5：测试与验证

1. 用项目指定 Python 运行 headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
2. 至少额外验证 package import：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; from retargeting.robot_pinocchio import RobotPinocchio"
   ```
3. 如可行，执行 editable install 后复测：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
4. 不启动 RViz、Open3D GUI、MuJoCo viewer、camera、real robot 或 ROS launch。

### Step 6：完成标准

- `src/retargeting/` 存在并包含第一批核心模块。
- 迁入模块使用 `retargeting.*` absolute imports。
- `pyproject.toml` 可让项目以 editable mode 安装。
- 至少一个低风险 console script 在 `pyproject.toml` 中声明，或明确说明本阶段为何延后具体脚本暴露。
- 旧路径文件仍能作为 import/脚本兼容层使用。
- `tests` 中 headless smoke tests 通过，或因缺少可选依赖产生清晰 skip/报告。

### 风险与处理

- 如果迁移全部模块导致 import 链扩大到 ROS/hardware/GUI 依赖，先缩小迁移范围，只搬测试覆盖的核心/offline 模块。
- 如果某些旧入口没有 `main()`，先只保留 re-export wrapper，不强行新增行为入口。
- 如果 editable install 暴露 package name 与旧 `src/__init__.py` 冲突，以 root `src/retargeting` 为新 source of truth，旧目录只做兼容。
- 如果测试依赖旧 `sys.path` 注入，更新 `tests/conftest.py` 让 repo root package 优先，同时必要时保留旧路径用于 wrapper 回归。

## Phase 1 Implementation Record

状态：已完成。

### 已完成的主要改动

1. 新建 root-level Python package：
   - `src/retargeting/__init__.py`
   - `src/retargeting/`

2. 迁入第一批核心/offline 模块：
   - `retarget_optimizer.py`
   - `robot_adaptor.py`
   - `robot_pinocchio.py`
   - `robot_benchmark.py`
   - `robot_mujoco.py`
   - `robot_teleoperation.py`
   - `single_hand_detector.py`
   - `offline_replay.py`
   - `retargeting_replay.py`
   - `vision_pro_detector.py`
   - `viser_retargeting_visualize.py`
   - `utils/`

3. 新 package 内部已改为 absolute package imports：
   - `from retargeting.robot_adaptor import RobotAdaptor`
   - `from retargeting.robot_pinocchio import RobotPinocchio`
   - `from retargeting.utils.utils_calc import ...`
   - `from retargeting.offline_replay import ...`

4. 更新 `pyproject.toml`：
   - 增加 `build-system`
   - 增加 `project` metadata
   - 使用 `setuptools` + `src` layout package discovery
   - 增加 console script：
     ```toml
     retargeting-replay = "retargeting.viser_retargeting_visualize:main"
     ```

5. 保留旧路径兼容 wrapper：
   - `ws_ros2/src/retargeting_benchmark/src/retarget_optimizer.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_adaptor.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_pinocchio.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_benchmark.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_mujoco.py`
   - `ws_ros2/src/retargeting_benchmark/src/robot_teleoperation.py`
   - `ws_ros2/src/retargeting_benchmark/src/single_hand_detector.py`
   - `ws_ros2/src/retargeting_benchmark/src/offline_replay.py`
   - `ws_ros2/src/retargeting_benchmark/src/retargeting_replay.py`
   - `ws_ros2/src/retargeting_benchmark/src/vision_pro_detector.py`
   - `ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py`

6. 增加旧路径兼容 helper：
   - `ws_ros2/src/retargeting_benchmark/src/_retargeting_compat.py`

   作用：旧路径脚本直接执行时，把 repo root 的 `src/` 插入 `sys.path`，使 wrapper 能导入新的 `retargeting` package。

7. 更新 tests：
   - `tests/conftest.py` 现在优先加入 root `src/`
   - 测试 import 从旧裸模块名改为 `retargeting.*`

### Viser 启动方式

新的 package 路径可以用 Python module 方式启动：

```bash
cd /home/ymr/mingrui/research/project_retargeting/retargeting

/home/ymr/miniconda3/envs/retargeting/bin/python \
  -m retargeting.viser_retargeting_visualize \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --hand-type leap \
  --end 100 \
  --port 8080
```

replay viewer 会重新运行 retarget optimizer。

### 后续可视化调整

按后续要求，viser viewer 已去除橙色机器人辅助可视化：

- 移除 `/current/robot/frames` 橙色点云
- 移除 `/current/robot/skeleton` 橙色连线
- 移除 `/trails/robot/*` 橙色轨迹线
- 移除 GUI 中的 `Show robot markers`
- 保留 `Show robot mesh`，URDF robot mesh 仍可显示

### 验证记录

1. package import 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; from retargeting.robot_pinocchio import RobotPinocchio"
   ```
   结果：通过。

2. 旧路径 wrapper import 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import sys; sys.path.insert(0, 'ws_ros2/src/retargeting_benchmark/src'); import robot_pinocchio; import offline_replay"
   ```
   结果：通过，解析到 `retargeting.robot_pinocchio` 和 `retargeting.offline_replay`。

3. editable install 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .
   ```
   结果：通过，成功安装 `retargeting-0.0.0`。

4. console script help 验证：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
   ```
   结果：通过。

5. headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
   结果：`9 passed`。

### 未纳入 Phase 1 的内容

- 未重写 optimizer API。
- 未配置化机器人参数。
- 未重排 ROS package、launch files、description package、assets 或 data。
- 未启动 RViz、Open3D GUI、MuJoCo viewer、camera、real robot 或 ROS launch。
