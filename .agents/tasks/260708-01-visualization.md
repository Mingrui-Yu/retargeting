
## Request 

我想要测试的时候有可视化。可以利用viser来可视化hand retargeting的实时结果。跑一段offline human hand trajectory，可视化retargeting后的trajectory，并且和human hand keypoint放在一起对比。现有codebase里应该有类似的功能，可能是使用rviz进行的可视化。请帮我找出这部分功能。

## 现有 RViz 可视化路径

当前 codebase 里已经有一条接近目标效果的 RViz 路径：

- 离线 AVP replay 加载逻辑在 `ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py`。
  - `flatten_stream_data()` 把记录时的 `stream` list 展平成 `stream_*` 数组。
  - `rebuild_stream_data()` 从 `.npz` 里的 `stream_*` 数组还原逐帧 Vision Pro stream dict。
  - `RobotTeleoperationMain.load_offline_data = True` 时，目前会 replay `data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz`。
- 逐帧 retargeting 在 `ws_ros2/src/retargeting_benchmark/src/robot_teleoperation.py` 的 `RobotTeleoperation.vision_pro_retarget()`。
  - 它解析一帧 Vision Pro stream。
  - 返回 `hand_kps_in_wrist`、`wrist_pose`、retargeted `qpos` 和误差指标。
  - 具体 retargeting objective 在 `hand_retarget()` 中实现。
- RViz 发布层集中在 `ws_ros2/src/retargeting_benchmark/src/rviz_visualize.py`。
  - `publish_hand_detection_results()` 发布 human hand keypoints、MANO 骨架线和 wrist frame。
  - `publish_robot_joint_states()` 发布 retarget 后的 robot joint states。
- RViz launch 文件在 `ws_ros2/src/retargeting_benchmark/launch/`。
  - `rviz_vis_paxini.py` 会为 Panda+Leap 启动 `robot_state_publisher`、`joint_state_publisher` 和 RViz。
  - 对应配置 `ws_ros2/src/retargeting_benchmark/rviz/vis_paxini.rviz` 订阅：
    - `/visualize/wrist_frame`
    - `/visualize/hand_keypoints`
    - `/visualize/hand_connections`
    - `/visualize/robot_description`
- Human hand 的拓扑和颜色在 `ws_ros2/src/retargeting_benchmark/src/utils/utils_mano.py`。
  - `MANO_LINE_PAIRS` 定义 hand skeleton。
  - `MANO_POINTS_COLORS` 定义 keypoint colors。
- Robot FK 可以通过 `ws_ros2/src/retargeting_benchmark/src/robot_pinocchio.py` 里的 `RobotPinocchio.get_frame_pose()` 获得。

## 相关数据

已有离线 replay 文件：

- `tests/fixtures/avp_short_replay.npz`
  - 2 帧。
  - 适合做 smoke test 和 headless validation。
  - 包含 `stream_*` 数组和 `retarget_qpos`。
- `data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz`
  - 760 帧。
  - 适合做真实长度的可视化 replay。
  - 包含同样的 stream layout，以及已记录的 `retarget_qpos`。

当前 `retargeting` conda env 中观察到的依赖状态：

- `pinocchio`：已安装。
- `nlopt`：已安装。
- `viser`：未安装。
- `avp_stream`：未安装。

现有测试已经通过 fake `avp_stream` 来测试离线解析，因为 `VisionProDetector.detect()` 本身只是纯 replay 解析，但 `vision_pro_detector.py` 在 module 顶层 import 了 `VisionProStreamer`。

## 实现计划

### 1. 抽取离线 Replay 工具

新增一个小的工具模块，例如：

`ws_ros2/src/retargeting_benchmark/src/offline_replay.py`

职责：

- 加载 `.npz` replay 文件。
- 从 `stream_*` 数组还原逐帧 stream dict。
- 提供 `--start`、`--end`、`--stride` 这类切片辅助。
- 逐帧重新运行 retargeting，生成 replay frame 数据。

这部分应复用 `main_robot_teleoperation.py` 中已有的 `rebuild_stream_data()` 行为，但把它从 ROS/RViz entrypoint 中移出来。

### 2. 让 Vision Pro 离线解析不依赖 `avp_stream`

当前问题：

- `vision_pro_detector.py` 在 module import 时就 import `VisionProStreamer`。
- 离线 replay 只需要 `VisionProDetector.detect()`。
- 如果没有安装 `avp_stream`，直接 import `vision_pro_detector` 会失败，除非像测试里一样 monkeypatch。

建议修法：

- 把纯 stream 解析逻辑移到独立函数，例如：
  - `parse_vision_pro_stream_frame(stream)`
- 让 `VisionProDetector.connect()` 成为唯一需要 `avp_stream` 的路径。
- 在 `connect()` 内部 lazy import `VisionProStreamer`。

这样离线 replay 和 viser 可视化就不需要 live Vision Pro 依赖。

### 3. 新增 Headless Retargeting Replay Runner

新增一个可复用 runner，例如：

`ws_ros2/src/retargeting_benchmark/src/retargeting_replay.py`

职责：

- 初始化 Panda+Leap 模型：
  - `RobotPinocchio(os.readlink("assets/panda_leap_paxini.urdf"), "urdf")`
  - `RobotAdaptor(... actuated_joints_name=[panda joints] + [joint_0..joint_15])`
  - `RobotControl(... use_hardware=False, use_virtual_hardware=False)`
  - `RobotTeleoperation(... input_device="vision_pro", mujoco_vis=False, use_real_hardware=False)`
- 对选中范围的第一帧：
  - 设置 robot initial wrist pose。
  - 解析 AVP initial wrist pose。
  - 调用 `set_robot_init_wrist_pose()` 和 `set_avp_init_wrist_pose()`。
- 对每一帧：
  - 解析 human hand keypoints 和 wrist pose。
  - 运行 `vision_pro_retarget()`，或使用文件中记录的 `retarget_qpos`。
  - 计算可视化需要的几何量：
    - world frame 下的 human keypoints。
    - robot wrist pose。
    - robot fingertip poses。
    - robot task link poses。
    - 可选的 trajectory trails。

CLI 参数：

- `--data tests/fixtures/avp_short_replay.npz`
- `--start 0`
- `--end -1`
- `--stride 1`
- `--fps 30`
- `--hand-type leap`

### 4. 新增 Viser 可视化 Backend

新增：

`ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py`

初始功能：

- 启动 `viser.ViserServer`。
- 将 human hand keypoints 画成小球或 point cloud。
- 用 `MANO_LINE_PAIRS` 画 human MANO skeleton。
- 画 human wrist coordinate frame。
- 画 robot retargeting 结果：
  - 第一版：用 Pinocchio FK 画 robot wrist 和 fingertip/task-link frames。
  - 用不同颜色画 robot fingertip positions 和 skeleton/task vectors。
- 画 trajectory trails：
  - human fingertip trails。
  - robot fingertip trails。
  - 可配置最近窗口长度。
- 加 viewer controls：
  - play/pause。
  - frame slider。
  - playback speed。
  - show/hide human。
  - show/hide robot。
  - show/hide trails。
  - reset frame/camera。

建议视觉约定：

- Human keypoints：亮蓝/青色 skeleton。
- Robot retargeted points/frames：橙色/红色。
- Wrist frames：RGB axes。
- Trajectory trails：低 alpha、较细线条。

### 5. Robot Mesh 支持

分两阶段实现：

Phase A：

- 只使用 Pinocchio FK frame positions。
- 可视化 robot wrist、fingertips、lower fingertip frames 和 objective vectors。
- 这种方式稳健，不依赖 viser 对 URDF mesh 的加载能力。

Phase B：

- 安装 `viser` 及必要 URDF/mesh 依赖后，为 Panda+Leap 加 URDF mesh rendering。
- 编码前按实际安装的 viser 版本确认准确 API。
- 如果直接 URDF 显示不方便，则手动解析 visual meshes，或使用兼容 helper library。

### 6. 测试

先加 headless tests：

- 加载 `tests/fixtures/avp_short_replay.npz`。
- 还原 stream frames。
- 解析一帧得到 `hand_kps_in_wrist` 和 `wrist_pose`。
- 把 human keypoints 转到 world coordinates。
- 用 Pinocchio 加载 Panda+Leap URDF。
- 把记录的 `retarget_qpos` 转成 full model qpos。
- 计算 robot wrist 和 fingertip FK。
- 断言 shape 正确、数值 finite。

可选 viser tests：

- 使用 `pytest.importorskip("viser")`。
- 尽量只 instantiate 或 dry-run backend helpers。
- 默认测试中不启动 GUI/RViz/Open3D/MuJoCo。

### 7. 集成注意事项

- 保持现有 RViz 路径不变。
- 将 viser 做成并行的 visualization backend。
- 不改变 retargeting objective 行为。
- 优先使用命令行参数，避免硬编码绝对路径。
- 默认执行保持 offline 和 no-hardware。

## 建议的第一个 Milestone

实现一个最小脚本，可以运行：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python \
  ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py \
  --data tests/fixtures/avp_short_replay.npz
```

预期结果：

- Viser server 启动。
- Frame slider 显示 2 帧。
- 可以看到 human hand keypoints 和 MANO skeleton。
- 可以看到 robot retargeted wrist/fingertip positions。
- Human 和 robot 点位在同一个 coordinate frame 下可对比。

然后扩展到：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python \
  ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --fps 30
```

## Job 实施记录

### 完成内容

- 已实现基于 `viser` 的 offline hand retargeting 可视化入口。
- 默认流程逐帧重新运行 retargeting，和 `main_robot_teleoperation.py` 中 `self.load_offline_data=True` 的核心流程一致。
- 已支持 human hand keypoints、MANO skeleton、human wrist frame。
- 已支持 robot retargeted frame markers、robot skeleton/task vectors、trajectory trails。
- 已支持 Panda+Leap robot mesh，可通过 `viser.extras.ViserUrdf` 从 URDF 加载 mesh，并随每帧 retargeted `qpos` 更新。
- viewer 中有以下交互开关：
  - `Playing`
  - `Frame`
  - `FPS`
  - `Show human`
  - `Show robot markers`
  - `Show robot mesh`
  - `Show trails`
  - `Trail length`

### 修改/新增文件

- `ws_ros2/src/retargeting_benchmark/src/vision_pro_detector.py`
  - 新增 `parse_vision_pro_stream_frame()`，把离线 Vision Pro stream 解析从 live `avp_stream` 依赖里拆出来。
  - `VisionProDetector.connect()` 中 lazy import `VisionProStreamer`。
  - `VisionProDetector.detect()` 复用新的纯解析函数。
- `ws_ros2/src/retargeting_benchmark/src/offline_replay.py`
  - 新增 `.npz` offline replay 加载工具。
  - 提供 `rebuild_stream_data()`、`load_offline_replay()`、`iter_frame_indices()`。
- `ws_ros2/src/retargeting_benchmark/src/retargeting_replay.py`
  - 新增 headless replay/FK 数据层。
  - 支持 Panda+Leap 和 Shadow 的基础配置。
  - 提供 `build_retarget_replay_frames()`，输出 human keypoints、wrist pose、retargeted `qpos`、robot FK poses。
  - 逐帧重新 retarget。
- `ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py`
  - 新增 viser viewer。
  - 支持 playback、frame slider、trails、human/robot markers、robot mesh。
  - 默认加载 robot mesh；可用 `--no-robot-mesh` 禁用。
- `ws_ros2/src/retargeting_benchmark/src/robot_teleoperation.py`
  - 将 `cv2` 改为 optional import。
  - 移除 offline retarget 路径不需要的顶层 `rclpy`、`tf2_ros`、`RobotControl`、`RobotMujoco`、`RobotPinocchio` 强依赖。
  - `RobotMujoco` 和 `RobotPinocchio` 改为使用时 lazy import。
  - 这样在没有 ROS/OpenCV 的 server 环境中也能运行 Vision Pro offline retarget。
- `tests/test_visualization_replay.py`
  - 新增 headless tests。
  - 覆盖 replay load、Vision Pro 离线解析、recorded-qpos FK、重新 retarget 分支。

### 安装依赖

安装到 `/home/ymr/miniconda3/envs/retargeting`：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install viser
/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install yourdfpy
```

安装后确认：

- `viser==1.0.30`
- `yourdfpy==0.0.60`

`viser` 用于 web-based 3D viewer；`yourdfpy` 用于 `ViserUrdf` 加载 URDF mesh。

### 最终推荐运行命令

长序列，逐帧重新 retarget，并显示 robot mesh：

```bash
cd /home/ymr/mingrui/research/project_retargeting/retargeting

/home/ymr/miniconda3/envs/retargeting/bin/python \
  ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --fps 30 \
  --port 8080
```

打开：

```text
http://localhost:8080
```

只看一小段：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python \
  ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --start 0 \
  --end 199 \
  --fps 30 \
  --port 8080
```

禁用 robot mesh，只看 markers：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python \
  ws_ros2/src/retargeting_benchmark/src/viser_retargeting_visualize.py \
  --data data/test_teleop/vision_pro/data_2025-01-16_20-27-43.npz \
  --no-robot-mesh
```

### 验证记录

运行 headless tests：

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest \
  tests/test_visualization_replay.py \
  tests/test_replay_smoke.py
```

结果：

```text
9 passed
```

额外验证：

- `py_compile` 通过。
- `viser` render dry-run 通过。
- 重新 retarget 分支通过。
- `ViserUrdf` 加载 Panda+Leap URDF mesh 通过。
- robot mesh 随 retargeted `qpos` 更新的 dry-run 通过。

### 注意事项

- 可视化逐帧调用 `RobotTeleoperation.vision_pro_retarget()` 重新 retarget。
- 可视化中 human 和 robot 之间的 global transform/offset 继承自原始 retargeting 代码：
  - AVP world 到 robot world 先绕 z 轴旋转 180 度。
  - 再加平移 `[0.7, 0.2, -1.0]`。
  - 第一帧还会按 robot initial wrist pose 和 AVP initial wrist pose 做相对 wrist 对齐。
- 当前 robot mesh 使用 Panda+Leap URDF：`assets/panda_leap_paxini.urdf` 指向的 URDF。
- 启动时可能会打印较多 joint list，这是 `RobotPinocchio.get_joint_index()` 的既有 debug print，不影响可视化功能。
