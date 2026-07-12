
## Initial Request

参看 /home/ymr/mingrui/research/project_retargeting/retargeting/temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md。现在我希望开始 Phase 4 的工作。

你可以参考之前完成的tasks（.agents/tasks/）

## Phase 4 Plan

目标：在 Phase 1 已完成 package 化、Phase 2 已完成第一批配置化、Phase 3 已拆清 core/ROS/hardware import 边界的基础上，清理 `assets/`、`data/`、`outputs/` 的职责边界。Phase 4 的核心完成标准是：offline/headless replay 不再依赖 symlink-based URDF 路径或源码树里的大实验输出；robot assets 以配置驱动的稳定路径加载；生成结果默认写入 gitignored output 目录；旧路径仍保留兼容 wrapper 或 alias，避免一次性破坏现有 ROS/实验脚本。

约束：

- 保持当前行为优先，Phase 0/1/2/3 的 headless tests 必须继续通过。
- 只做 data/assets/output 边界整理，不重写 optimizer API，不做 README 全面重写，不启动 ROS、RViz、camera、real robot、Open3D GUI 或 MuJoCo viewer。
- 不删除大数据或实验结果；先通过 `.gitignore`、迁移说明和兼容路径降低源码污染风险。若需要真正移动/删除大文件，必须单独确认。
- `retargeting` core 继续保持无 ROS import；asset path resolver 不能引入 ROS package 依赖。
- 优先让 `configs/robots/*.yaml` 成为 robot asset 的单一入口。旧脚本中的硬编码 `assets/*.urdf` / `os.readlink(...)` 只能逐步收敛到配置或兼容 helper。
- 所有验证默认使用 `/home/ymr/miniconda3/envs/retargeting/bin/python`，并优先运行 headless/offline tests。

### Phase 4 范围

本阶段处理：

1. `assets/` 目录结构和 path resolver。
2. `configs/robots/*.yaml` 中 URDF/MJCF/mesh path 的稳定化。
3. `data/` 中 demo fixture、test fixture、实验输出、teleop recording 的边界。
4. `outputs/` 默认输出目录和 `.gitignore` 规则。
5. 依赖 `assets/*.urdf` symlink、`os.readlink(...)`、`data/simulation/*`、`data/teleop_process/*` 的 headless 可测调用点。
6. 最小文档或 task record，说明新旧路径兼容策略和推荐路径。

本阶段不处理：

- optimizer typed API 和 objective 拆分。
- 大规模 ROS package/description package 重命名。
- 真实硬件 topic/IP 配置全面整理。
- GUI/hardware/manual replay 的端到端验证。
- 全量 benchmark dataset 的下载、发布、压缩或远端托管。

### Step 1：盘点并冻结现状

先建立资产和数据现状清单，避免后续移动时漏掉隐式依赖：

- 列出 `assets/` 下现有 URDF symlink、MJCF/XML、mesh 文件。
- 列出 `data/` 下哪些文件被 tests/configs/CLI 默认路径引用，哪些是实验产物或历史输出。
- 用 `rg` 扫描这些模式：
  - `assets/`
  - `data/`
  - `os.readlink`
  - `.npz`
  - `.urdf`
  - `.xml`
- 标出调用点类别：
  - core/headless replay 必须稳定。
  - ROS/teleop/hardware 旧入口暂时兼容。
  - plotting/experiment scripts 可以改默认 output path，但不作为 Phase 4 主验证路径。

完成后在 implementation record 中记录迁移前的兼容目标，避免把“整理目录”变成不可追踪的行为改变。

### Step 2：引入 asset path resolver

在 `src/retargeting/config/` 或 `src/retargeting/assets/` 增加轻量 path resolver，用于统一解析 repo-relative asset path：

- 接受 repo-relative path、absolute path、以及旧 symlink path。
- 默认以项目根目录为 base，避免依赖当前工作目录。
- 对 symlink path 做兼容解析，但新配置不再要求 `path_is_symlink: true`。
- 校验文件存在时返回明确错误，错误信息包含原始配置路径、解析后的路径、当前工作目录。
- 不 import ROS、MuJoCo、viser、Open3D。

建议最小 API：

```python
resolve_repo_path(path: str | Path, *, base_dir: Path | None = None) -> Path
resolve_asset_path(path: str | Path, *, follow_symlink: bool = true) -> Path
```

如果 Phase 2 的 `RobotModelConfig.resolved_path()` 已经覆盖一部分逻辑，本阶段优先扩展它，而不是另起一套并行规则。

### Step 3：规范 robot asset 目录，但保留旧入口

目标目录建议：

```text
assets/
├── robots/
│   ├── panda_leap_paxini/
│   │   ├── urdf/
│   │   └── meshes/
│   ├── panda_leap_tac3d/
│   │   ├── urdf/
│   │   ├── mjcf/
│   │   └── meshes/
│   └── panda_shadow/
│       ├── urdf/
│       └── meshes/
└── scenes/
```

实施策略要分两步：

1. 先把 `configs/robots/*.yaml` 改到新路径，并让 tests 覆盖新路径加载。
2. 再保留 `assets/panda_leap_paxini.urdf`、`assets/panda_leap_tac3d.urdf`、`assets/panda_shadow.urdf` 等旧路径作为兼容 symlink 或 wrapper，直到旧 ROS/experiment scripts 全部迁完。

URDF 来源策略：

- Phase 4 不强制删除 `ws_ros2/src/my_robot_description/urdf/*.urdf`。
- 新 core/offline 配置应直接指向稳定的 repo asset path，不要求运行时 `os.readlink("assets/*.urdf")`。
- ROS description package 仍可保留自己的 xacro/urdf 组织；Phase 4 只保证 Python core 不依赖 ROS package layout。

### Step 4：迁移 headless 路径中的 symlink/readlink 依赖

优先处理可测试、影响最大的调用点：

- `src/retargeting/config/schema.py`
- `src/retargeting/retargeting_replay.py`
- `src/retargeting/viser_retargeting_visualize.py`
- `src/retargeting/robot_teleoperation.py` 中 headless/offline 可覆盖的默认路径
- `tests/test_replay_smoke.py`
- `tests/test_config_loading.py`

原则：

- 新代码从 `RobotConfig.model.resolved_path()` 或统一 resolver 获取 robot model path。
- Tests 不再断言 `assets/*.urdf` 必须是 symlink；改为断言配置路径存在、可解析、Pinocchio 可加载。
- 旧路径 wrapper 可以继续用 `os.readlink(...)`，但 core/default replay 不应要求它。
- 对 `__main__` demo block 和未纳入测试的实验脚本，只做低风险路径替换或留下 TODO，不在本阶段重构行为。

### Step 5：整理 data / examples / fixtures 边界

推荐边界：

```text
tests/fixtures/
└── avp_short_replay.npz          # pytest 必需的小 fixture

examples/data/
└── avp_short_replay.npz          # 用户 quickstart 可选 demo；可与 tests fixture 同源或通过说明复用

data/
└── README.md                     # 说明 full dataset / benchmark data 不随源码维护

outputs/                         # gitignored
├── teleop/
├── simulation/
├── benchmark/
└── plots/
```

实施策略：

- 保留 `tests/fixtures/avp_short_replay.npz` 作为测试 source of truth。
- `configs/apps/replay_avp.yaml` 继续默认指向 tiny fixture，保证 fresh checkout 可以跑 headless smoke。
- 新增或更新 `.gitignore`，让 `outputs/`、teleop recordings、simulation generated `.npz`、plot artifacts 默认不进入 git。
- 不直接删除 `data/teleop_process/`、`data/simulation/`、`data/test_teleop/`；只把新默认输出迁到 `outputs/`，并记录旧数据属于历史/外部实验产物。
- 如需要提供用户 demo 数据，可优先让 `examples/data/` 复用 tiny fixture，而不是复制大文件。

### Step 6：把新输出默认写入 `outputs/`

将可控脚本的默认保存路径从 `data/...` 改到 `outputs/...`：

- teleop recording 默认：`outputs/teleop/{timestamp}/data.npz`
- simulation/benchmark 默认：`outputs/simulation/...` 或 `outputs/benchmark/...`
- plot 默认：`outputs/plots/...`

优先改已有配置或 CLI 参数默认值；不要为了改路径而重写脚本结构。旧命令如果显式传入 `data/...`，仍应继续工作。

### Step 7：新增 Phase 4 测试

新增或更新 headless tests，覆盖：

1. 所有 `configs/robots/*.yaml` 的 model path 都能通过 resolver 找到真实文件。
2. `RobotPinocchio` 能从配置解析后的 URDF path 加载模型，不依赖 `os.readlink("assets/*.urdf")`。
3. `configs/apps/replay_avp.yaml` 指向 tiny fixture，且 replay smoke test 仍能生成帧。
4. `.gitignore` 包含 `outputs/` 和新默认输出路径。
5. 静态扫描 core/default replay 路径，确认关键模块不再硬依赖旧 symlink 断言。

测试命令：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

如果新增专项测试，先单独运行：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py
```

### Step 8：兼容性验证

最低验证集：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -c "from retargeting.config import load_robot_config; print(load_robot_config('configs/robots/panda_leap_paxini.yaml').model.resolved_path())"
```

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

```bash
/home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
```

如果 Phase 4 改动了 CLI 默认数据路径，再补充：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml --no-robot-mesh --end 3
```

这个命令可能启动 viser server，但不需要 ROS/hardware；若当前环境不适合启动服务，则只运行构建 replay frames 的 headless tests。

### Step 9：完成标准

Phase 4 完成时应满足：

- `configs/robots/*.yaml` 指向稳定、可解析的 robot asset path。
- Core/offline replay 默认路径不依赖 `assets/*.urdf` symlink 或当前工作目录。
- 旧 `assets/*.urdf` 路径仍兼容，不破坏旧 ROS/实验脚本。
- `tests/fixtures/` 保留小 replay fixture；新默认输出进入 `outputs/` 或明确 gitignored 路径。
- `.gitignore` 覆盖生成结果，不再鼓励把 teleop/simulation/plot 输出作为源码维护。
- Phase 0/1/2/3 headless tests 继续通过。
- Implementation record 记录：
  - 实际迁移的 asset/data 路径。
  - 保留的旧兼容路径。
  - 未迁移的 ROS/hardware/experiment 脚本。
  - 验证命令和结果。

### 风险与处理

- URDF mesh path 风险：移动 URDF 后 mesh 相对路径可能失效。先用 Pinocchio load tests 验证；必要时保留 URDF 原位置并只更新配置解析层，不强行搬 mesh。
- ROS description package 风险：ROS launch 依赖 `package://my_robot_description/...`。Phase 4 不改 ROS package layout，只让 Python core 脱离 symlink。
- 大数据误删风险：本阶段不删除历史 `data/` 内容；只调整默认路径和 ignore 规则。
- 旧脚本路径风险：`ws_ros2/src/retargeting_benchmark/src/*` 仍可能硬编码旧路径。优先保证旧路径兼容，后续 Phase 再迁移高耦合 runtime entrypoints。
- 测试 fixture 复制风险：避免把同一 demo 数据复制多份；若需要 `examples/data/`，优先用说明或小文件同步策略，并在 record 中写清楚来源。

## Phase 4 Implementation Record

状态：已完成。

### 已完成的主要改动

1. 新增规范化 robot asset 目录：
   - `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf`
   - `assets/robots/panda_leap_tac3d/urdf/panda_leap_tac3d.urdf`
   - `assets/robots/panda_shadow/urdf/panda_shadow.urdf`
   - `assets/robots/panda_leap_tac3d/mjcf/panda_leap_tac3d.xml`
   - `assets/robots/panda_leap_tac3d/mjcf/panda_leap_tac3d_asset.xml`
   - `assets/scenes/scene.xml`

   URDF 文件复制自 `ws_ros2/src/my_robot_description/urdf/*.urdf`。URDF 内部的 relative mesh path 通过 `urdf/` 下的相对 symlink 保持可解析：
   - `panda -> ../../../panda`
   - `leap_hand -> ../../../leap_hand`
   - `shadow_hand -> ../../../../ws_ros2/src/my_robot_description/urdf/shadow_hand`

2. 保留旧 root-level asset 兼容入口：
   - `assets/panda_leap_paxini.urdf`
   - `assets/panda_leap_tac3d.urdf`
   - `assets/panda_shadow.urdf`
   - `assets/panda_leap_tac3d.xml`
   - `assets/panda_leap_tac3d_asset.xml`
   - `assets/scene.xml`

   本阶段没有删除旧 symlink 或旧 XML，避免破坏 ROS/实验脚本。

3. 扩展 config path resolver：
   - `src/retargeting/config/io.py` 新增 `resolve_asset_path(...)`
   - `src/retargeting/config/__init__.py` 导出 `resolve_asset_path`
   - `src/retargeting/config/schema.py` 新增 `RobotModelConfig.resolved_path()`

   新 resolver 支持：
   - repo-relative path
   - absolute path
   - 旧 symlink path
   - 旧 symlink target 相对 repo root 的兼容解析

4. 更新 robot configs，使默认 model path 指向新 asset layout：
   - `configs/robots/panda_leap_paxini.yaml`
   - `configs/robots/panda_leap_tac3d.yaml`
   - `configs/robots/panda_shadow.yaml`

   `path_is_symlink` 已设为 `false`，`robot_config.robot_file_path` 现在直接返回稳定 URDF 文件路径。

5. 更新 core demo/default path，减少对旧 symlink 的硬依赖：
   - `src/retargeting/robot_adaptor.py`
   - `src/retargeting/robot_pinocchio.py`
   - `src/retargeting/robot_teleoperation.py`

   这些 demo block 现在通过 `load_robot_config("configs/robots/panda_leap_tac3d.yaml")` 获取 URDF 和 joints。

6. 更新旧 ROS teleop 入口的默认 robot model 加载：
   - `ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py`

   该文件仍是 ROS/hardware runtime 入口，但默认 URDF 和 actuated joints 改为从 `retargeting.config` 获取，不再直接 `os.readlink("assets/*.urdf")`。

7. 更新默认输出目录：
   - teleop recording 默认从 `data/teleop_process/...` 改为 `outputs/teleop/...`
   - offline quantitative result 默认从 `data/simulation/shadow/complex_8.npz` 改为 `outputs/simulation/shadow/complex_8.npz`
   - plot output 默认从 `data/experiments/plot` 改为 `outputs/plots`

   已改文件：
   - `ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py`
   - `ws_ros2/src/retargeting_benchmark/src/plot_success_rate.py`
   - `ws_ros2/src/retargeting_benchmark/src/plot_completion_time.py`

8. 更新 `.gitignore`：
   - `outputs/`
   - `outputs/teleop/`
   - `outputs/simulation/`
   - `outputs/benchmark/`
   - `outputs/plots/`
   - `data/teleop_process/`
   - `data/simulation/**/*.npz`
   - `data/experiments/plot/`

9. 新增 `data/README.md`，说明：
   - `data/` 只作为本地实验数据/历史数据位置。
   - headless tests 和默认 replay config 不依赖 `data/` 下的大文件。
   - 新生成结果应写入 `outputs/`。

10. 更新并新增测试：
    - `tests/test_config_loading.py`
    - `tests/test_replay_smoke.py`
    - `tests/test_phase4_assets_data.py`

    覆盖：
    - 所有 robot config 使用 `assets/robots/...` 新路径。
    - 新路径和旧 symlink 都能通过 resolver 解析。
    - Pinocchio 从 config 解析后的 URDF 加载。
    - replay app config 使用 `tests/fixtures/avp_short_replay.npz`，不依赖 `data/`。
    - `.gitignore` 覆盖 Phase 4 输出目录。

### 验证记录

1. Robot config path 解析：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "from retargeting.config import load_robot_config; [print(load_robot_config(p).robot_file_path) for p in ['configs/robots/panda_leap_paxini.yaml','configs/robots/panda_leap_tac3d.yaml','configs/robots/panda_shadow.yaml']]"
   ```
   结果：三份 config 均解析到 `assets/robots/.../urdf/*.urdf`。

2. Phase 4 专项测试：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_phase4_assets_data.py
   ```
   结果：`4 passed`。

3. Config + replay smoke tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_config_loading.py tests/test_replay_smoke.py
   ```
   结果：`9 passed`。

4. 完整 headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
   结果：`19 passed`。

5. CLI help：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
   ```
   结果：通过，可见 `--config`、`--robot`、`--retarget`、`--data` 等参数。

6. 旧 teleop 脚本语法检查：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m py_compile ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py
   ```
   结果：通过。

7. 新旧 asset resolver 兼容检查：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "from retargeting.config import resolve_asset_path; print(resolve_asset_path('assets/panda_leap_paxini.urdf')); print(resolve_asset_path('assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf'))"
   ```
   结果：旧 symlink 解析到 `ws_ros2/src/my_robot_description/urdf/panda_leap_paxini.urdf`，新路径解析到 `assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf`。

### 未纳入 Phase 4 的内容

- 未删除或迁移历史 `data/teleop_process/`、`data/simulation/`、`data/test_teleop/` 数据。
- 未重排 ROS description package，也未修改 ROS launch 中的 `package://my_robot_description/...` 资源组织。
- 未启动 ROS、RViz、camera、real robot、Open3D GUI 或 MuJoCo viewer。
- 未重写 optimizer typed API。
- 未全面清理所有 plotting/experiment scripts 的 input `base_dir`，只调整了默认 plot output path。
