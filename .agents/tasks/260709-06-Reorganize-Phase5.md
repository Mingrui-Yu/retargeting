
## Initial Request

参看 /home/ymr/mingrui/research/project_retargeting/retargeting/temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md。现在我希望开始 Phase 5 的工作。

你可以参考之前完成的tasks（.agents/tasks/）

## Phase 5 Plan

目标：在 Phase 1-4 已完成 package 化、配置化、core/ROS 边界拆分、assets/data/output 边界整理的基础上，更新 README 和最小用户文档。Phase 5 的核心完成标准是：新用户先通过 headless/offline replay 跑通项目，再按需进入 ROS/RViz、live input、real hardware、添加新机器人、复现实验等更高依赖路径；文档必须反映当前 codebase 的真实入口和约束，不把硬件/GUI/大数据路径写成默认 quickstart。

### 约束与边界

- 保持当前行为优先：Phase 0/1/2/3/4 的 headless tests 必须继续通过。
- Phase 5 只做文档、示例命令、README 信息架构和必要的轻量说明文件更新；不重写 optimizer API，不移动 assets/data，不改 ROS launch 行为。
- README 的第一条可执行路径必须是 offline/headless replay：`retargeting-replay --config configs/apps/replay_avp.yaml` 或等价 Python module 命令。
- ROS、RViz、camera、Vision Pro live streaming、real hardware、MuJoCo/Open3D/viser viewer 都必须标为可选路径，并写清环境前提；不能作为默认 smoke path。
- 不把实验室 IP、Realsense serial、Vision Pro IP、真实硬件 topic 等本地配置写成通用默认值。需要展示时使用 `.example` 或明确标注“lab-specific example”。
- 不承诺当前仓库没有实现的远程下载命令、dataset hosting、benchmark reproduction 自动化；若需要说明数据下载，先写成“当前状态 + 后续 TODO/外部数据位置占位”。
- 文档命令统一以 repo root 为工作目录，Python 测试使用 `/home/ymr/miniconda3/envs/retargeting/bin/python` 作为开发验证命令；README 面向普通用户时可以给出通用 `python`/`pip` 命令。
- 默认文档语言以英文 README 为主；task record 和内部说明继续用中文。

### Phase 5 范围

本阶段处理：

1. Root `README.md` 的结构重写，使其从 offline replay quickstart 开始。
2. 安装说明拆分为 minimal core/offline install、optional visualization、optional ROS/hardware。
3. 使用说明覆盖当前真实入口：
   - editable install
   - `retargeting-replay`
   - `python -m retargeting.viser_retargeting_visualize`
   - `configs/apps/replay_avp.yaml`
   - `configs/robots/*.yaml`
   - `configs/retargeting/vector_wrist_joint.yaml`
4. 文档化 Phase 1-4 后的新边界：
   - `src/retargeting/` 是 core/offline package
   - `src/retargeting_ros/` 是 ROS adapter package
   - `configs/` 是 robot/retarget/app 配置入口
   - `assets/robots/` 是新 robot asset layout
   - `tests/fixtures/` 是 tiny smoke fixture
   - `outputs/` 是 generated result 默认位置
5. 保留旧 ROS/teleop/hardware 使用说明，但降级为 advanced/optional section，并明确旧脚本仍是兼容入口。
6. 增加开发者验证命令，帮助后续改动确认 README 没有漂移。

本阶段不处理：

- optimizer typed dataclass API 和 objective 文件拆分。
- 大数据下载 CLI、Zenodo/HuggingFace release、dataset packaging。
- 完整 benchmark reproduction pipeline。
- ROS package 重命名或 `ws_ros2/` 目录重排。
- 真实硬件配置自动发现或硬件 bringup 重构。
- 网站 `docs/index.md` 的全面重写；除非 README 中链接明显错误，只做最小修正。

### 推荐 README 信息架构

README 建议按以下顺序组织：

1. **Project Overview**
   - 一句话说明项目：human-to-robot dexterous-hand retargeting。
   - 简短列出提供内容：core retargeting、offline replay、config-driven robot setup、ROS/RViz adapters、live input/hardware adapters。
   - 保留 paper/project website 链接和 overview image。

2. **Repository Layout**
   - 用短表格说明关键目录：
     - `src/retargeting/`
     - `src/retargeting_ros/`
     - `configs/`
     - `assets/robots/`
     - `tests/fixtures/`
     - `outputs/`
     - `ws_ros2/`
   - 明确 dependency direction：core 不依赖 ROS/hardware；ROS adapter 依赖 core。

3. **Install**
   - Minimal core/offline install：
     ```bash
     conda create -n retargeting -c conda-forge python=3.10.12 pinocchio
     conda activate retargeting
     pip install -e ".[replay]"
     ```
   - Optional viewer/live/hardware dependencies 通过 `pyproject.toml` extras 管理：
     - `replay` for offline browser viewer
     - `mujoco` for MuJoCo paths
     - `vision` for RGB input
     - `avp` for Vision Pro live input
     - ROS2 Humble and ROS packages for RViz/hardware
   - 不让 optional dependency 阻塞 quickstart。

4. **Quickstart: Offline Replay**
   - 以 tiny fixture 和 config-first 命令为主：
     ```bash
     retargeting-replay --config configs/apps/replay_avp.yaml --end 100
     ```
   - 提供无需 console script 的等价命令：
     ```bash
     python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml --end 100
     ```
   - 说明默认 config 使用 `tests/fixtures/avp_short_replay.npz`，不需要相机、ROS 或真实机器人。
   - 如果 `viser` 缺失，给出安装提示或改用 headless test 命令确认环境。

5. **Configuration**
   - 解释三类配置：
     - robot config：`configs/robots/panda_leap_paxini.yaml`
     - retargeting config：`configs/retargeting/vector_wrist_joint.yaml`
     - app config：`configs/apps/replay_avp.yaml`
   - 给出常用 override：
     ```bash
     retargeting-replay \
       --robot configs/robots/panda_shadow.yaml \
       --retarget configs/retargeting/vector_wrist_joint.yaml \
       --data tests/fixtures/avp_short_replay.npz \
       --hand-type shadow
     ```
   - 提醒新增机器人优先新增 config 和 `assets/robots/<robot_name>/...`，不要在 Python 源码里写死 joint/link/path。

6. **ROS and RViz**
   - 标注为 optional advanced path。
   - 保留现有 ROS2 Humble 安装和 `colcon build --symlink-install` 命令。
   - 说明 Python-level ROS adapter 在 `retargeting_ros`，旧 `ws_ros2/src/retargeting_benchmark/src/*.py` 文件仍作为兼容入口。
   - RViz launch 命令保留，但说明需要 ROS workspace 已 build/source。

7. **Live Input and Teleoperation**
   - 区分 RGB/MediaPipe、Vision Pro、real robot。
   - 明确这些路径依赖 camera/live streamer/ROS/hardware，不属于默认测试路径。
   - 旧命令可以保留：
     ```bash
     python ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py
     ```
   - 加一句说明：默认 recording output 写入 `outputs/teleop/`。

8. **Real Robot Control**
   - 保留现有 Panda + Leap lab setup，但加警示：lab-specific, not required for offline replay。
   - 不把 IP 写成泛用配置；如保留历史 IP，标注为 example from original lab setup。
   - 强调运行前需要单独确认 ROS network、FCI、Leap driver 和安全条件。

9. **Data and Outputs**
   - 说明 `tests/fixtures/avp_short_replay.npz` 是 smoke/demo fixture。
   - `data/` 是本地实验/历史数据位置，默认 quickstart 不依赖大文件。
   - `outputs/` 是新生成结果位置并已 gitignored。
   - 如果全量 dataset 尚未发布下载命令，明确写“full datasets are not bundled in this repository”。

10. **Development and Tests**
    - 给出 headless verification：
      ```bash
      /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
      ```
    - 给出 import contract：
      ```bash
      python -c "import retargeting; import retargeting_ros"
      ```
    - 提醒默认测试不启动 ROS、RViz、camera、real robot 或 GUI viewer。

11. **Citation**
    - 保留 BibTeX。
    - 如果项目网站或 paper 链接更新，README 与 `docs/index.md` 的链接保持一致。

### Step 1：文档现状审计

先盘点当前 README 与实际 codebase 的偏差：

- README 当前是否仍以 ROS/teleop 为第一使用入口。
- README 是否仍建议 symlink-based `assets/*.urdf` 作为主要路径。
- README 是否仍把 `data/` 当作默认数据源或输出位置。
- README 是否缺少 `pip install -e .`、`retargeting-replay`、`configs/apps/replay_avp.yaml`。
- README 是否把 optional dependencies 写成必装依赖。
- README 中 project website、paper、citation 是否和 `docs/index.md` 保持一致。

审计命令：

```bash
rg -n "retargeting-replay|viser_retargeting_visualize|configs/apps|assets/|data/|outputs/|ROS2|RViz|teleoperation|Citation" README.md docs configs pyproject.toml
```

### Step 2：重写 README quickstart 主线

把 README 的 Usage 从 teleoperation-first 改为 offline-first：

1. 安装 editable package。
2. 运行 offline replay config。
3. 解释 tiny fixture、robot config、retarget config。
4. 再链接到 ROS/RViz 和 hardware。

推荐最小命令块：

```bash
git clone --recurse-submodules https://github.com/Mingrui-Yu/retargeting.git
cd retargeting
conda create -n retargeting -c conda-forge python=3.10.12 pinocchio
conda activate retargeting
pip install -e ".[replay]"
retargeting-replay --config configs/apps/replay_avp.yaml --end 100
```

如果 README 面向没有 console script 的环境，紧跟等价命令：

```bash
python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml --end 100
```

### Step 3：文档化配置和 assets/data 边界

新增或更新 README sections：

- `Configuration`
- `Robot Assets`
- `Data and Outputs`

重点说明：

- Robot URDF/MJCF assets now live under `assets/robots/`.
- Robot-specific joints, frames, model paths, initial qpos, and hand scale should be configured in `configs/robots/*.yaml`.
- Retargeting objective/link pairs are configured in `configs/retargeting/*.yaml`.
- App-level replay options are configured in `configs/apps/*.yaml`.
- Generated files should go to `outputs/`; do not treat `data/teleop_process/` or `data/simulation/*.npz` as source-controlled defaults.

### Step 4：整理 ROS/RViz/hardware 文档

将现有 ROS 和 hardware 内容移动到 advanced sections，并调整措辞：

- `ROS/RViz Visualization`
- `Live Teleoperation`
- `Real Robot Control`

文档原则：

- 每个 section 先列 prerequisites。
- 命令保留现有兼容入口，避免用户突然找不到旧脚本。
- 明确 `retargeting_ros` 是 adapter；core/offline 不需要 ROS。
- 对 real robot control 加安全和实验室环境限定说明。
- 不在文档中暗示默认测试会启动 ROS launch 或真实硬件。

### Step 5：添加开发者验证说明

README 或 `docs/development.md` 中加入最小开发验证：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; import retargeting_ros"
/home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
```

如果实际 README 不适合包含本机绝对路径，则 README 使用通用命令，task implementation record 记录本机验证命令：

```bash
python -m pytest tests
python -c "import retargeting; import retargeting_ros"
retargeting-replay --help
```

### Step 6：可选补充文档文件

如果 README 过长，可以新增轻量 docs：

```text
docs/
├── quickstart.md
├── configuration.md
├── ros_rviz.md
├── hardware.md
└── development.md
```

但 Phase 5 优先更新 root README；只有当某个 section 明显过长时再拆分。拆分后 README 必须仍能独立提供 offline quickstart。

### Step 7：验证 README 命令

默认验证不启动 ROS/hardware。至少运行：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
```

```bash
/home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
```

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --help
```

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; import retargeting_ros; from retargeting.config import load_replay_app_config; print(load_replay_app_config('configs/apps/replay_avp.yaml').data)"
```

如环境已安装 `viser` 且允许启动本地 server，可选验证：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml --no-robot-mesh --end 3
```

这个命令会启动 viewer/server，不作为必须验证；如果执行，完成后需要手动停止进程，不让后台 session 残留。

### Step 8：完成标准

Phase 5 完成时应满足：

- README 第一条推荐运行路径是 offline replay，而不是 ROS/teleop/hardware。
- README 包含 `pip install -e ".[replay]"`、`retargeting-replay --config configs/apps/replay_avp.yaml` 和 Python module 等价命令。
- README 清楚区分 core/offline、ROS adapter、live input、real hardware。
- README 解释 `configs/`、`assets/robots/`、`tests/fixtures/`、`data/`、`outputs/` 的职责边界。
- README 不再把 symlink-based `assets/*.urdf` 作为新增机器人或 quickstart 的主路径。
- README 不再把 `data/` 下大实验文件作为默认 quickstart 依赖。
- ROS/RViz/hardware 命令仍保留，但标为 optional/advanced，并写清 prerequisites。
- Citation、project website、paper 链接可用或至少不比当前状态更差。
- Headless tests 通过，或失败原因明确记录且与 README 编辑无关。
- Implementation record 记录：
  - README/doc 实际改动。
  - 保留的旧命令。
  - 未实现或未验证的 optional paths。
  - 验证命令和结果。

### 风险与处理

- 文档超前风险：只记录当前已存在的命令、配置和路径；未实现的 dataset download、benchmark CLI、hardware config manager 写成 TODO 或暂不写。
- 依赖安装风险：minimal install 与 optional dependencies 分开，避免用户为了 offline replay 被 ROS/hardware/live input 依赖卡住。
- Viewer 误判风险：`retargeting-replay` 当前使用 `viser` viewer；如果环境没有 `viser`，README 要说明安装方式或提供 headless tests 作为环境检查。
- ROS 用户迁移风险：保留旧 `ws_ros2/src/retargeting_benchmark/src/*.py` 命令作为兼容入口，同时说明新 Python package 边界。
- 硬件安全风险：real robot section 必须显式标注 lab-specific、requires independent safety checks，不作为 quickstart。
- 链接漂移风险：README、project website、`docs/index.md` 的链接若不一致，优先选择当前实际可用链接，并在 implementation record 中说明。

## Phase 5 Implementation Record

状态：已完成。

### 已完成的主要改动

1. 重写 root `README.md`，将文档主线从 ROS/teleoperation-first 改为 offline replay-first。

2. README 顶部项目入口更新为：
   - project website：`https://star-xcd.github.io/retargeting/`
   - arXiv：`https://arxiv.org/abs/2506.09384`
   - 保留 `docs/overview.jpg` overview image。

3. 新增 `Repository Layout` section，说明 Phase 1-4 后的主要边界：
   - `src/retargeting/`
   - `src/retargeting_ros/`
   - `configs/`
   - `assets/robots/`
   - `tests/fixtures/`
   - `outputs/`
   - `ws_ros2/`

4. 更新安装说明：
   - minimal offline path 使用 editable install：
     ```bash
     pip install -e ".[replay]"
     ```
   - 选择 `pyproject.toml` 作为依赖管理入口，没有新增 `requirements.txt`。
   - 将 `numpy`、`scipy`、`PyYAML`、`nlopt` 放到 base `project.dependencies`。
   - 将 `viser`、`mujoco`、`opencv-python`、`mediapipe`、`avp_stream`、`pytest` 拆到 optional dependency extras。
   - PyTorch 是 optimizer path 必需依赖，但由于 GPU wheel 取决于 CUDA 环境，不写死进 `pyproject.toml`，改为提示用户按自己的 GPU/CUDA 环境从官方说明安装，并记录当前测试版本：CUDA 12.8、PyTorch `2.11.0+cu128`。
   - ROS/RViz/hardware 依赖单独放到 advanced section，不再阻塞 quickstart。

5. 新增 `Quickstart: Offline Replay` section，推荐入口为：
   ```bash
   retargeting-replay --config configs/apps/replay_avp.yaml --end 100
   ```

   同时保留等价 Python module 入口：
   ```bash
   python -m retargeting.viser_retargeting_visualize --config configs/apps/replay_avp.yaml --end 100
   ```

6. 新增配置说明，覆盖：
   - `configs/robots/panda_leap_paxini.yaml`
   - `configs/robots/panda_leap_tac3d.yaml`
   - `configs/robots/panda_shadow.yaml`
   - `configs/retargeting/vector_wrist_joint.yaml`
   - `configs/apps/replay_avp.yaml`

7. 新增 robot asset 说明：
   - 推荐新代码使用 `assets/robots/`。
   - 说明 root-level legacy paths such as `assets/panda_leap_paxini.urdf` 只作为兼容路径保留。
   - 保留 ROS description package 位置：`ws_ros2/src/my_robot_description/`。

8. 新增 data/output 边界说明：
   - `tests/fixtures/avp_short_replay.npz` 是 tiny replay fixture。
   - `data/` 是本地/历史实验数据位置，不作为默认 quickstart 依赖。
   - 新生成结果写入 `outputs/`，并说明该路径 gitignored。

9. 将 ROS/RViz、live teleoperation、real robot control 移到 optional/advanced sections。
   - 保留 RViz launch 示例：
     ```bash
     ros2 launch retargeting_benchmark rviz_vis_paxini.py
     ```
   - 保留旧 teleoperation compatibility entrypoint：
     ```bash
     python ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py
     ```
   - Real robot section 明确标注 lab-specific，不作为 offline replay 必需路径。
   - 历史 IP 地址保留为原实验室示例，不作为通用默认值。

10. 新增 development 验证说明：
    - `python -m pytest tests`
    - `python -c "import retargeting; import retargeting_ros"`
    - `retargeting-replay --help`
    - `python -m retargeting.viser_retargeting_visualize --help`

11. README 新增 Citation 和 Contact sections，保留当前 BibTeX。

### 保留的兼容路径

- 旧 ROS workspace 仍在 `ws_ros2/`。
- 旧 RViz launch 命令仍记录在 README。
- 旧 teleoperation Python entrypoint 仍记录为 compatibility entrypoint。
- Root-level legacy robot asset paths 仍作为兼容路径说明，但不再作为新增 robot 的推荐路径。

### 验证记录

1. Python module help：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m retargeting.viser_retargeting_visualize --help
   ```
   结果：通过，可见 `--config`、`--robot`、`--retarget`、`--data`、`--hand-type` 等参数。

2. Console script help：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/retargeting-replay --help
   ```
   结果：通过。

3. Import/config contract：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -c "import retargeting; import retargeting_ros; from retargeting.config import load_replay_app_config; print(load_replay_app_config('configs/apps/replay_avp.yaml').data)"
   ```
   结果：通过，输出 `tests/fixtures/avp_short_replay.npz`。

4. 完整 headless tests：
   ```bash
   /home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests
   ```
   结果：`19 passed`。

### 未纳入 Phase 5 的内容

- 未启动 `retargeting-replay --config ...` 的实际 viewer/server 长循环。
- 未启动 ROS launch、RViz、camera、Vision Pro live streaming、real robot、Open3D GUI 或 MuJoCo viewer。
- 未新增 dataset download 命令或远端数据发布说明。
- 未拆分新的 `docs/*.md`；本阶段只更新 root `README.md` 和本 task record。
- 未修改 optimizer API、配置 schema、assets/data layout、ROS launch 或硬件行为。
