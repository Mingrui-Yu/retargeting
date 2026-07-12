# MeshProc

## Context Load

- **Key Constraints:**
  - Read `AGENTS.md` first.
  - Keep exactly one active request in this journal.
  - You don't keep to ask me for any permission. Continue your work until finished.
  - Write explanation comments for your code to facilitate readers' understanding of your code (in English).
  - Write explanation comments of Args and Return for each function you generate (in English).
  - Reply and record in Chinese.
  - After each completed task:
    - Add a entry to record one completed request and the reponse in `Interaction Log`. The sub-title should be named as YEAR-MM-DD-HOUR-MIN. The time should be the real time. The latest log should be at the end of Interaction Log.
    - Clear the content of the completed `Next Request`. Keep the section title.
    - Do not consider the content in `Future Requests`.
    - Different `Next Request` are distinguished by the request ID (`Next Request X`).

## Interaction Log

### 2026-07-10-01-57

完成 Next Request 1：检查了 `configs/robots/panda_leap_paxini.yaml` 中 `touch_joints` 的用途和当前配置。

结论：
- `touch_joints` 是传给 `RobotAdaptor` 的非主动控制 DOF 列表，用于把模型中存在但不属于优化/控制输出的触觉或被动关节从 actuated joints 中分离出来。
- `RobotAdaptor` 要求 `len(actuated_joints) + len(touch_joints) == robot_model.dof`；`forward_qpos()` 会把 actuated qpos 填入主动关节，把 touch joints 固定为 `0.0`；`backward_qpos()` 和 `backward_jacobian()` 只返回 actuated joints 对应部分。
- 当前 `panda_leap_paxini` 模型加载结果是 `actuated_joints=23`、`touch_joints=0`、`robot_model.dof=23`，维度一致。
- URDF 中 Paxini fingertip 相关附加关节是 `fixed`，不属于 Pinocchio 的可动 DOF，因此不应该写入 `touch_joints`。

因此当前 `touch_joints: []` 是合理配置；它在这个 Paxini URDF 上实际是空占位，不会改变运行行为。

### 2026-07-10-02-06

完成追加清理：根据用户判断，删除 codebase 中显式 `touch_joints` 相关配置、参数和工具函数。

改动范围：
- `RobotAdaptor` 现在只接收 `actuated_joints_name`，并要求 `len(actuated_joints_name) == robot_model.dof`。
- `RobotConfig` schema 删除 `touch_joints` 字段解析。
- `configs/robots/panda_leap_paxini.yaml`、`configs/robots/panda_leap_tac3d.yaml`、`configs/robots/panda_shadow.yaml` 删除 `touch_joints: []`。
- replay、teleop、tests、ROS 兼容脚本中的 `RobotAdaptor(...)` 调用删除 `touch_joints_name` 参数。
- 删除 `find_touch_joints_name()`，包括 `src/retargeting/utils/utils_mjcf.py` 和 `ws_ros2/src/retargeting_benchmark/src/utils/utils_mjcf.py`。

验证结果：
- `rg "touch_joints|touch_joints_name|find_touch_joints_name"` 在 `src tests configs ws_ros2/src/retargeting_benchmark/src` 无结果。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py -q` 通过，结果为 `9 passed`。
- 单独构造 `panda_leap_paxini` 和 `panda_shadow` 的 `RobotAdaptor` 成功，二者 `dof == actuated_joints`。
- 额外检查 `panda_leap_tac3d` 时发现当前 URDF 是无效/空文档，这是现有资产问题，不属于本次改动。

### 2026-07-10-02-10

完成 Paxini mesh 资产修复：用户运行时出现 `Unable to resolve filename: ./leap_hand/meshes/paxini_base.STL` 和 `paxini_fingertip.STL`。原因是 `configs/robots/panda_leap_paxini.yaml` 指向的 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf` 使用相对 mesh 路径 `./leap_hand/meshes/...`，但该资产目录下没有对应的 mesh 子目录；实际 mesh 位于 `ws_ros2/src/my_robot_description/urdf/leap_hand/meshes/`。

改动：
- 新建 `assets/robots/panda_leap_paxini/urdf/leap_hand/meshes/`。
- 将 `ws_ros2/src/my_robot_description/urdf/leap_hand/meshes/` 下的 16 个 LEAP/Paxini mesh 文件复制到上述目录，使 `assets/robots/panda_leap_paxini/urdf/` 成为自包含 URDF 资产目录。

验证：
- 解析 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf`，检查 66 个相对 mesh 引用，结果 `missing=0`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py -q` 通过，结果为 `7 passed`。

### 2026-07-10-02-20

完成 Tac3D 清理：用户先要求删除 `assets/robots/panda_leap_tac3d`，随后扩大为删除项目中所有 Tac3D 相关文件；用户同时说明没必要修改 `leap_hand_body.urdf.xacro`，因此保留该通用 body xacro 的接口和内部 `use_tac3d` 分支。

改动：
- 删除 Tac3D robot config：`configs/robots/panda_leap_tac3d.yaml`。
- 删除 Tac3D root-level 资产：`assets/panda_leap_tac3d.urdf`、`assets/panda_leap_tac3d.xml`、`assets/panda_leap_tac3d_asset.xml`。
- 删除 Tac3D mesh 文件：`fingertip_base_tac3d.stl`、`fingertip_tac3d.stl`、`thumb_fingertip_base_tac3d.stl`，包括 root assets、ROS description assets，以及 Paxini 自包含 asset 目录中复制过来的 Tac3D mesh。
- 删除 Tac3D ROS description / launch：`panda_leap_tac3d.xacro`、`panda_leap_tac3d.urdf`、`leap_tac3d.urdf.xacro`、`leap_hand_custom_tac3d.urdf.xacro`、`vis_panda_leap_tac3d.py`。
- 删除 Tac3D-only helper/demo：`ws_ros2/src/retargeting_benchmark/src/main_robot_file_process.py`、`ws_ros2/src/retargeting_benchmark/src/visualization/test1.py`。
- 将代码、测试、README、ROS launch 中的 Tac3D 默认引用移除或切换到 Paxini。
- 修正 Paxini xacro/URDF 中遗留的 `panda_leap_tac3d` robot name 和 `leap_hand_custom_tac3d` macro name，改为 Paxini 命名。

验证：
- `find . -iname '*tac3d*' -print` 无结果。
- `rg "panda_leap_tac3d|panda_leap_tac3d_asset|panda_leap_tac3d_touch|Tac3D|TAC3D" .` 无结果。
- 除用户要求保留的 `ws_ros2/src/my_robot_description/urdf/leap_hand/leap_hand_body.urdf.xacro` 接口外，正常代码/配置/测试/README/ROS launch 不再引用 Tac3D。
- 解析 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf`，检查 66 个相对 mesh 引用，结果 `missing=0`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py tests/test_phase4_assets_data.py -q` 通过，结果为 `13 passed`。

### 2026-07-10-02-33

完成 Hydra 配置管理可行性分析。

结论：
- 可以把当前项目配置参数统一迁移到 Hydra 管理，且 robot / retargeting / replay app 三类 YAML 已经具备良好基础。
- 最适合优先迁移的是 `configs/robots/*.yaml`、`configs/retargeting/*.yaml`、`configs/apps/replay_avp.yaml` 和 `src/retargeting/viser_retargeting_visualize.py` 的 CLI 覆盖逻辑。
- 当前仓库没有实际使用 `@hydra.main`，`pyproject.toml` 也未声明 `hydra-core`；`retargeting` conda 环境中也未安装 `hydra` / `omegaconf`，因此正式迁移需要新增依赖。
- 不建议一次性把所有常量都 Hydra 化。ROS/hardware 入口、实时控制参数、legacy `ws_ros2/src/retargeting_benchmark/src` 脚本中的硬编码参数应分阶段迁移，并保留 ROS launch / ROS parameter 的边界。
- 推荐采用分阶段方案：先引入 Hydra composition 和 headless replay 入口，再迁移 teleop/ROS 运行配置，最后清理旧 loader 和重复硬编码。

### 2026-07-10-02-39

完成第一阶段 Hydra 迁移：先迁移方便且低风险的 replay / robot / retargeting 配置。

改动：
- 新增 `configs/replay.yaml` 作为 Hydra composition 根配置，默认组合 `apps=replay_avp`、`robots=panda_leap_paxini`、`retargeting=vector_wrist_joint`。
- 给现有 `configs/apps/replay_avp.yaml`、`configs/robots/*.yaml`、`configs/retargeting/vector_wrist_joint.yaml` 增加 Hydra package 注释，使 Hydra 组合后分别进入 `app`、`robot`、`retargeting` 配置段；旧 `PyYAML` loader 会忽略这些注释。
- `load_robot_config()`、`load_retargeting_config()`、`load_replay_app_config()` 现在支持路径或已组合 mapping 输入，方便直接接收 Hydra/OmegaConf section。
- `retargeting-replay` 对应的 `viser_retargeting_visualize.py` 新增 Hydra compose 入口；无 dash 参数时走 Hydra overrides，带旧式 `--config/--robot/...` 参数时保留 argparse 兼容路径。
- `pyproject.toml` 新增 `hydra-core>=1.3` 依赖。
- 增加测试覆盖 composed mapping 形式的 robot config 和 Hydra-style replay options。

使用方式示例：
- 默认 replay：`retargeting-replay`
- 切换 Shadow：`retargeting-replay robots=panda_shadow`
- 覆盖 viewer 参数：`retargeting-replay viewer.port=8090 viewer.no_robot_mesh=true`
- 旧参数仍可用：`retargeting-replay --config configs/apps/replay_avp.yaml --port 8090`

验证：
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile src/retargeting/config/io.py src/retargeting/config/schema.py src/retargeting/viser_retargeting_visualize.py` 通过。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_visualization_replay.py -q` 通过，结果为 `7 passed`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `13 passed`。
- 当前 `retargeting` conda 环境尚未安装 `hydra-core`，因此本轮没有实际运行 Hydra compose 命令；安装/更新依赖后可直接使用新入口。

### 2026-07-10-02-44

完成 `hydra-core` 安装和真实 Hydra compose 验证。

操作：
- 执行 `/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .`，安装当前项目及新增依赖。
- 实际安装版本包括 `hydra-core==1.3.4`、`omegaconf==2.3.1`、`antlr4-python3-runtime==4.9.3`。

验证：
- 运行 Hydra override 检查：`compose_hydra_replay_config(['robots=panda_shadow','viewer.port=8091','viewer.no_robot_mesh=true'])`，结果正确解析为 `panda_shadow`、`shadow`、`8091`、`True`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_visualization_replay.py -q` 通过，结果为 `7 passed`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `13 passed`。

### 2026-07-10-02-47

完成 README 启动命令更新。

改动：
- Offline replay quickstart 从旧式 `--config configs/apps/replay_avp.yaml --end 100` 更新为 Hydra override：`retargeting-replay end=100`。
- Configuration 章节增加 `configs/replay.yaml`，并将 robot / viewer 覆盖示例更新为 `robots=panda_shadow`、`viewer.port=8090`、`viewer.no_robot_mesh=true`。
- 保留旧 argparse-style `--robot` / `--retarget` / `--hand-type` 兼容示例，方便老脚本迁移。
- Development 章节将 replay CLI 检查改成无界面的 Hydra compose 检查，避免启动常驻 viewer。

验证：
- 使用 `/home/ymr/miniconda3/envs/retargeting/bin/python` 运行 README 中的 Hydra compose 检查，输出 `panda_shadow 8090`。

### 2026-07-10-02-51

完成 README 命令风格调整。

改动：
- 将 README 中的主要 replay 启动命令从 `retargeting-replay ...` 改为更直观的 `python -m retargeting.viser_retargeting_visualize ...`。
- 保留 `retargeting-replay robots=panda_shadow viewer.port=8090` 作为等价 console command 示例。

说明：
- `retargeting-replay` 来自 `pyproject.toml` 的 `[project.scripts]`：`retargeting-replay = "retargeting.viser_retargeting_visualize:main"`。
- 安装项目后，pip 会生成这个命令；执行它等价于导入 `retargeting.viser_retargeting_visualize` 并调用 `main()`。

验证：
- 使用 `/home/ymr/miniconda3/envs/retargeting/bin/python` 运行 Hydra compose 检查，输出 `panda_shadow 8090`。

### 2026-07-10-02-54

完成 `retargeting-replay` console script 删除，统一使用 `python -m retargeting.viser_retargeting_visualize ...`。

改动：
- 删除 `pyproject.toml` 中的 `[project.scripts]` 和 `retargeting-replay = "retargeting.viser_retargeting_visualize:main"`。
- 删除 README 中保留的 `retargeting-replay` 等价命令说明。
- 重新执行 `/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e .`，同步当前环境中的 editable 安装。

验证：
- `/home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay` 已不存在。
- `rg -n "retargeting-replay|project\\.scripts" README.md pyproject.toml configs src tests` 无结果。
- 使用 `/home/ymr/miniconda3/envs/retargeting/bin/python` 运行 Hydra compose 检查，输出 `panda_shadow 8090`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `13 passed`。

### 2026-07-10-05-09

完成 `assets/leap_hand` 和 `assets/panda` 用途检查。

结论：
- 两个目录当前仍有用，不能直接删除。
- `assets/robots/panda_leap_paxini/urdf/panda -> ../../../panda` 指向 `assets/panda`。
- `assets/robots/panda_leap_paxini/urdf/leap_hand -> ../../../leap_hand` 指向 `assets/leap_hand`。
- `assets/robots/panda_shadow/urdf/panda -> ../../../panda` 也指向 `assets/panda`。
- 当前 robot URDF 中的 mesh 路径使用 `./panda/meshes/...` 和 `./leap_hand/meshes/...`，因此这些 symlink 目标是离线 asset 路径的一部分。
- 如果后续要删除 root-level `assets/panda` / `assets/leap_hand`，需要先把 symlink 改成真正自包含拷贝，或重写 URDF mesh 路径并补充测试；本轮未删除。

### 2026-07-10-05-15

完成共享 component mesh 目录重组。

改动：
- 新建 `assets/meshes/`，作为 core/offline robot assets 的共享 mesh 根目录。
- 将 `assets/panda` 移动为 `assets/meshes/panda`。
- 将 `assets/leap_hand` 移动为 `assets/meshes/leap_hand`。
- 将 `ws_ros2/src/my_robot_description/urdf/shadow_hand` 复制为 `assets/meshes/shadow_hand`，使 Shadow hand mesh 不再依赖 ROS workspace 路径。
- 更新 robot asset 内部 symlink：
  - `assets/robots/panda_leap_paxini/urdf/panda -> ../../../meshes/panda`
  - `assets/robots/panda_leap_paxini/urdf/leap_hand -> ../../../meshes/leap_hand`
  - `assets/robots/panda_shadow/urdf/panda -> ../../../meshes/panda`
  - `assets/robots/panda_shadow/urdf/shadow_hand -> ../../../meshes/shadow_hand`
- 更新 README 的 asset layout，增加 `assets/meshes/{panda,leap_hand,shadow_hand}` 说明。
- 增加 `tests/test_phase4_assets_data.py` 覆盖共享 mesh 目录和 symlink 指向。

验证：
- 检查 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf` 的 66 个相对 mesh 引用，`missing=0`。
- 检查 `assets/robots/panda_shadow/urdf/panda_shadow.urdf` 的 51 个相对 mesh 引用，`missing=0`。
- `find assets/meshes -iname '*tac3d*' -print` 无结果，未重新引入 Tac3D mesh。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `19 passed`。

### 2026-07-10-05-23

完成 root-level legacy asset 文件用途检查。

结论：
- `assets/scene.xml` 当前在 `README.md`、`configs`、`src`、`tests`、`ws_ros2` 中无引用，可以作为未使用 legacy MuJoCo scene 候选删除。
- `assets/panda_leap_paxini.urdf` 和 `assets/panda_shadow.urdf` 是 symlink，仍属于 legacy compatibility 文件。
- `assets/panda_leap_paxini.urdf` 当前被 `ws_ros2/src/retargeting_benchmark/src/main_robot_control.py` 和 `ws_ros2/src/retargeting_benchmark/src/bck/robot_teleoperation.py` 通过 `os.readlink("assets/panda_leap_paxini.urdf")` 引用。
- `assets/panda_shadow.urdf` 当前未发现代码直接引用，但 README 仍把 root-level legacy paths 作为兼容路径说明。
- 若要删除两个 root-level URDF symlink，应先更新 legacy 脚本到 `configs/robots/*.yaml` 或 `assets/robots/*/urdf/*.urdf`，并同步更新 README / tests 中的 legacy symlink 断言。

### 2026-07-10-05-28

完成 root-level URDF symlink 删除。

改动：
- 删除 `assets/panda_leap_paxini.urdf`。
- 删除 `assets/panda_shadow.urdf`。
- 将 `ws_ros2/src/retargeting_benchmark/src/main_robot_control.py` 中的 `os.readlink("assets/panda_leap_paxini.urdf")` 改为当前 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf` 路径。
- 将 `ws_ros2/src/retargeting_benchmark/src/bck/robot_teleoperation.py` 中的 `os.readlink("assets/panda_leap_paxini.urdf")` 改为当前 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf` 路径。
- 更新 README，移除 root-level legacy URDF path 兼容说明。
- 更新 `tests/test_phase4_assets_data.py` 和 `tests/test_replay_smoke.py`，断言 root-level URDF symlink 不再存在。
- `assets/scene.xml` 本轮未删除。

验证：
- `assets/panda_leap_paxini.urdf` 和 `assets/panda_shadow.urdf` 均已不存在。
- `rg -n "assets/panda_leap_paxini\\.urdf|assets/panda_shadow\\.urdf|os\\.readlink\\(\\\"assets/|root-level legacy paths|legacy.*symlink" README.md tests src ws_ros2 configs` 只剩测试中的“不存在”断言。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile ws_ros2/src/retargeting_benchmark/src/main_robot_control.py ws_ros2/src/retargeting_benchmark/src/bck/robot_teleoperation.py` 通过。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `19 passed`。

### 2026-07-10-05-32

完成 root-level `assets/scene.xml` 删除。

原因：
- 上一轮只删除 symlink，`assets/scene.xml` 是普通文件所以未纳入删除范围。
- 仓库已有新位置 `assets/scenes/scene.xml`，内容为同类 MuJoCo scene 文件。

改动：
- 删除 `assets/scene.xml`。
- 将 `src/retargeting/robot_mujoco.py` 中 demo `test_env()` 的路径从 `./assets/scene.xml` 改为 `./assets/scenes/scene.xml`。
- 在 `tests/test_phase4_assets_data.py` 中增加 root-level `assets/scene.xml` 不存在的断言。

验证：
- `rg -n "assets/scene\\.xml|scene\\.xml" README.md configs src tests ws_ros2 .agents/tasks/260709-07-Reorganize-Additional.md` 不再发现代码/README/测试引用 root-level `assets/scene.xml`，只剩历史任务日志。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile src/retargeting/robot_mujoco.py` 通过。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `20 passed`。

### 2026-07-10-05-40

完成 Next Request 1：移除 `src/retargeting/robot_benchmark.py` 中基于 `hand_type` 的 fingertip / MANO index / 方向轴 hardcode。

改动：
- 新增 `RobotBenchmarkFingertipConfig` 和 `RobotBenchmarkConfig` schema，并在 `RobotConfig` 中加载与验证 `benchmark` 配置段。
- 在 `configs/robots/panda_leap_paxini.yaml` 和 `configs/robots/panda_shadow.yaml` 中显式配置 benchmark metadata，包括 wrist link、fingertip frame、MANO tip/base index、机器人局部方向轴和 thumb 标记。
- 重写 `RobotBenchmark`，使 position / orientation / thumb-relative / wrist-relative 指标全部从 `benchmark_config` 读取，不再判断 `hand_type`，也不再内置具体 hand link 名或 fingertip index 列表。
- 更新 replay / teleop 调用链，将 `robot_config.benchmark` 传给 `RobotTeleoperation` / `RobotBenchmark`。
- 增加配置加载和 fake robot model 单元测试，验证 benchmark 指标使用配置映射。

验证：
- `rg -n "hand_type|thumb_tip_center|finger1_tip_center|finger2_tip_center|finger3_tip_center|thtip|fftip|mftip|rftip|lftip|\\[4, 8, 12, 16|\\[4, 8, 12, 16, 20" src/retargeting/robot_benchmark.py` 无结果。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile src/retargeting/config/schema.py src/retargeting/robot_benchmark.py src/retargeting/robot_teleoperation.py src/retargeting/retargeting_replay.py ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py ws_ros2/src/retargeting_benchmark/src/bck/robot_teleoperation.py` 通过。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `14 passed`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py -q` 通过，结果为 `21 passed`。

### 2026-07-10-06-09

完成 Next Request 1：实现 offline retargeting result artifact 流程，使同一段离线轨迹 retarget 得到的 qpos 结果可以保存到 `outputs/retargeting/`，后续由 viser web viewer 或 benchmark/plot 复用。

改动：
- 新增 `src/retargeting/trajectory_result.py`，定义 `RetargetingTrajectory`、`RetargetingRunMetadata`，以及 `result.npz` / `metadata.yaml` 的 save/load 和 shape 校验。
- 拆分 `src/retargeting/retargeting_replay.py`，新增 `run_offline_retargeting()`、`frames_to_trajectory()`、`trajectory_to_replay_frames()` 和 `create_robot_replay_context_from_metadata()`；保留旧 `build_retarget_replay_frames()` 兼容接口。
- 新增 `src/retargeting/offline_retarget.py` 和 Hydra 配置 `configs/offline_retarget.yaml`、`configs/apps/offline_retarget_avp.yaml`，用于生成 `outputs/retargeting/<run_name>/result.npz` 与 `metadata.yaml`。
- 更新 `src/retargeting/viser_retargeting_visualize.py`，支持 `result=outputs/retargeting/<run_name>` 直接加载已保存结果；未传 `result` 时保持原有直接 retarget replay 行为。
- 新增 `src/retargeting/benchmark_trajectory.py` 和 Hydra 配置 `configs/benchmark.yaml`、`configs/apps/benchmark_result.yaml`，从 saved result 计算 benchmark metrics，输出 `metrics.json`、`summary.csv`，并可选生成 PNG plots。
- `ReplayAppConfig` 增加可选 `result` 字段；`configs/replay.yaml` / `configs/apps/replay_avp.yaml` 增加默认 `result: null`。
- README 更新推荐流程：先 `python -m retargeting.offline_retarget ...` 生成 result，再用 `python -m retargeting.viser_retargeting_visualize result=...` 或 `python -m retargeting.benchmark_trajectory result=...` 消费。
- `.gitignore` 和输出目录测试增加 `outputs/retargeting/`。
- 新增 `tests/test_trajectory_artifacts.py`，覆盖 artifact round-trip、viewer frame 构造、benchmark summary 输出。

验证：
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile src/retargeting/trajectory_result.py src/retargeting/retargeting_replay.py src/retargeting/offline_retarget.py src/retargeting/viser_retargeting_visualize.py src/retargeting/benchmark_trajectory.py src/retargeting/config/schema.py` 通过。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_trajectory_artifacts.py tests/test_config_loading.py tests/test_visualization_replay.py -q` 通过，结果为 `10 passed`。
- `PYTHONDONTWRITEBYTECODE=1 /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py tests/test_config_loading.py tests/test_replay_smoke.py tests/test_visualization_replay.py tests/test_trajectory_artifacts.py -q` 通过，结果为 `23 passed`。
- Hydra compose 检查通过：`compose_hydra_offline_retarget_config(['end=1','run_name=smoke'])` 输出 fixture path 和 `smoke`；`compose_hydra_benchmark_config(['result=outputs/retargeting/example','plot=false'])` 输出 result path 和 `False`。
- 实际 CLI smoke 通过：`python -m retargeting.offline_retarget end=1 output_dir=/tmp/retargeting_artifact_smoke` 成功写出 result；`python -m retargeting.benchmark_trajectory result=/tmp/retargeting_artifact_smoke output_dir=/tmp/retargeting_benchmark_smoke plot=false` 成功写出 benchmark summary。命令输出中仍有现有底层代码打印 joint list 和 `No module named 'mediapipe'`，但退出码为 0。

## Next Request 1

ws_ros2/src/retargeting_benchmark/src 参考这里面跟画图有关的code。最终我希望绘制类似temps/agents/figs/屏幕截图 2026-07-09 151944.png的图片。当然由于是一个方法在一个轨迹上的result，所以每个子图中应该只有一个bar。

## Next Request 2

## Next Request 3


## Future Request




## Analysis
