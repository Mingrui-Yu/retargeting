# Headless Regression Test Context

Run tests from the repository root with the project Python environment. Default tests must not start ROS, RViz, cameras, Vision Pro live streaming, a real robot, Open3D GUI, or a MuJoCo viewer.

The focused offline regression file is `tests/test_replay_smoke.py`. It currently covers:

- the Phase 4 robot-asset layout, including removal of obsolete root-level URDF links;
- the promoted 760-frame AVP fixture `tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz` and its first recorded `retarget_qpos`;
- shared `decode_avp_sample()` parsing without the live AVP dependency;
- configured Panda+Leap URDF loading and `RobotAdaptor` qpos/Jacobian mappings;
- Panda+Leap profile lower-bound overrides reaching both the optimizer and its solver;
- one `VectorWristJointOptimizer.retarget()` call for each supported SLSQP backend: NLopt and SciPy.

The full headless suite also validates the self-contained Panda+Leap portable bundle manifest and relative resource
paths. When the optional `mujoco` dependency is installed, it compiles the bundled MJCF without launching a viewer;
otherwise that test reports an explicit skip. The shared execution-flow tests additionally validate joint-name state
mapping, atomic backend periods, position-actuator control ranges, 20 Hz command stepping, startup interpolation,
direct execution, missing-detection hold behavior, mapping/reset state, and the absence of viewer or saved-trajectory
dependencies. The offline-human tests verify that AVP online/offline acquisition shares one decoder, only raw
`stream_*` arrays are loaded, existing `retarget_qpos` data is ignored, hold/direct frames consume one command period,
startup frames may consume multiple periods, and real raw AVP input retargets directly into headless MuJoCo.
Passive mjviser tests use fake
server/scene objects to cover initial and per-frame publication, client/final-state waits, and guaranteed cleanup
without opening a socket or browser. They also bind all 23 Panda+Leap hinge-joint readouts through compiled
`jnt_qposadr` values and verify atomic updates do not modify `data.qpos`.

Focused command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_replay_smoke.py -q
```

Focused MuJoCo runtime command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_mujoco_backend.py tests/test_mujoco_runtime.py tests/test_mujoco_offline_simulation.py tests/test_mjviser_live.py -q
```

Full headless regression command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests -q
```

The baseline verified on 2026-07-17 with the `mujoco` extra installed is 8 focused replay tests, 11 focused MuJoCo
tests, and 75 tests in the full suite. Update this document when the intended test scope changes; optional
dependencies should produce explicit skips rather than a behavior change.

## Reorganization Phase 0 Baseline

The pre-change baseline was rerun on 2026-07-17 in the `retargeting` environment:

- `tests/test_replay_smoke.py`: 8 passed in 2.42 s.
- focused MuJoCo backend and online/offline runtime tests: 11 passed in 3.76 s.
- full headless suite: 75 passed in 9.18 s.

The following behavior and numeric contracts are frozen for the reorganization:

- `avp_teleop_2025-01-16_20-27-43.npz` has SHA-256
  `c9f9ff03cd62db0a49f7fc2183a599edf168f93073ad14381d00123f14e7de72` and contains 760 wrist frames,
  760 finger frames, and a finite `(760, 23)` qpos trajectory. Its first qpos remains pinned in
  `test_avp_teleop_replay_fixture_shape_and_expected_qpos` with `rtol=0` and `atol=1e-12`.
- One 20 Hz command advances exactly `0.05 s`, using 25 physics steps at `0.002 s` per step.
- With speed limits `[1.0, 2.0]`, a request for `[1.0, -1.0]` produces first command `[0.05, -0.1]` and reaches
  `[1.0, -1.0]` after 20 frames; simulation and realtime clocks both read `1.0 s` with zero overrun.
- A missing detection holds the existing `[0.2, -0.2]` command while still advancing simulation by `0.05 s`.
- Three aligned offline-human frames process indices `[0, 1, 2]` and advance `0.15 s`. One invalid pre-alignment
  frame followed by one valid frame still advances `0.1 s`. The real two-frame raw AVP path advances `0.1 s`
  without reading stored `retarget_qpos` data.

`tests/test_package_import_boundaries.py` adds three Phase 0 contracts. It parses imports with Python AST, requires
zero forbidden runtime imports from `retargeting/core`, freezes the 15 known package-level migration edges, and
verifies isolated `import retargeting` while AVP, OpenCV, MediaPipe, MuJoCo, Viser, Open3D, and ROS imports are
blocked. The migration allowlist must only shrink and must be empty in the final architecture. With these tests,
the post-change full headless suite is 78 passed in 9.27 s.

## Reorganization Phase 1 Baseline

Phase 1 adds seven domain/config boundary contracts:

- `HandObservation` and `HandInput` have one canonical implementation in `retargeting.core.types`; transitional
  `retargeting.inputs` imports resolve to the same objects.
- `retargeting.config.core` contains the robot kinematics, retargeting objective/profile, and solver schemas.
  `RobotConfig` has no simulator model and `RetargetingProfileConfig` has no command policy.
- The unchanged profile YAML loads independently as a core profile and `TeleoperationCommandConfig`, preserving
  all 23 Panda+Leap command limits and the existing `profile.teleoperation.*` Hydra values.
- The unchanged robot YAML loads independently as a core URDF config and `MujocoRobotBindingConfig`, preserving
  the existing MJCF path and 20 Hz simulator behavior.
- Artifact metadata retains the legacy `simulation_model` and `teleoperation` nodes during the migration even
  though those fields are no longer owned by the core dataclasses.
- AST boundary checks forbid `retargeting/core` from importing the mixed `retargeting.config` facade or migration
  schema; core modules import `retargeting.config.core` directly.

The Phase 1 verification on 2026-07-17 is 33 focused optimizer/solver/replay/artifact/MuJoCo tests, 42 focused
config/domain/import tests, and 85 tests in the full headless suite. The final full suite completed in 9.59 s;
`compileall` and `git diff --check` also passed.

## Reorganization Phase 2 Baseline

Phase 2 adds four runtime ownership contracts and moves all optional execution adapters out of `retargeting`:

- AVP, RGB, the offline-human loader, and `HandObservationAdapter` now live under `teleoperation.inputs`.
- `RobotBackend` and the lazy headless MuJoCo backend now live under `teleoperation.backends`.
- Detection, teleoperation mode/output/command, robot-control, MuJoCo timing, and robot-simulator binding schemas and
  loaders now have the single canonical owner `teleoperation.config`.
- The old `retargeting.inputs`, `retargeting.backends`, and singular `teleoperation.input` packages are removed;
  tests and ROS compatibility entrypoints use only canonical paths.
- Importing `teleoperation.inputs`, `teleoperation.backends`, or `teleoperation.config` succeeds while AVP,
  OpenCV, MediaPipe, and MuJoCo imports are blocked. Optional implementations remain lazy.
- `src/retargeting` has zero direct MuJoCo, AVP, OpenCV, or MediaPipe imports. Its 15 temporary AST edges now consist
  only of app/pipeline imports of `teleoperation` plus Viser imports that move in Phase 3/4; the edge budget did not
  grow from Phase 0.

The Phase 2 verification on 2026-07-17 is 4 focused runtime ownership tests, 49 focused config/domain/architecture
tests, 33 focused optimizer/solver/replay/artifact/MuJoCo tests, and 89 tests in the full headless suite. The final
full suite completed in 12.64 s; `compileall` and `git diff --check` also passed.

## Reorganization Phase 3 Baseline

Phase 3 establishes `retargeting_apps` as the only application composition package:

- CLI dispatch, app runners, offline and benchmark pipelines, trajectory artifacts, and replay visualization moved
  from `retargeting` to `retargeting_apps` without compatibility wrappers at the old paths.
- Replay and viewer schemas moved from `retargeting.config` to `retargeting_apps.config`; the core config facade no
  longer exports application schemas.
- The public console command remains `retargeting`, now backed by `retargeting_apps.main:main`. Hydra app ids,
  override keys, and config files are unchanged, and module examples use `python -m retargeting_apps.main`.
- The Phase 0/2 migration allowlist for `src/retargeting` is now empty. `retargeting` has zero imports of
  `teleoperation`, `retargeting_apps`, `retargeting_ros`, optional viewer/runtime modules, or ROS modules.
- Importing `retargeting_apps.main` remains lazy while AVP, OpenCV, MediaPipe, MuJoCo, Open3D, and Viser imports are
  blocked. `teleoperation` imports neither `retargeting_apps` nor `retargeting_ros`.
- Artifacts remain NumPy/YAML data rather than pickle or Python-class serialization, so saved metadata does not
  encode the moved module paths.

The Phase 3 verification on 2026-07-17 is 40 focused config/domain/import/ownership tests, 16 focused
artifact/replay tests, 11 focused MuJoCo backend and runtime tests, and 94 tests in the full headless suite. The full
suite completed in 12.58 s. The editable install and generated `retargeting` console script were refreshed and
verified to resolve `retargeting_apps.main:main`; `compileall`, package discovery, and `git diff --check` passed.
`ruff` and `black` remain unavailable in the environment, so those checks were not run.

## Reorganization Phase 4 Baseline

Phase 4 separates offline retargeting computation, raw-human runtime policy, and app orchestration:

- `retargeting.core.sequence.retarget_observation_sequence` accepts only canonical `HandObservation` values, a core
  solver protocol, and optional temporal state. It returns `RetargetingResult` values and has no file, detector,
  progress, viewer, app, or teleoperation dependency.
- `teleoperation.avp_alignment.initialize_avp_alignment` is the single AVP wrist-alignment helper shared by live,
  offline, artifact, and compatibility paths.
- `AlignedMujocoTeleoperationDriver` owns pre-alignment hold and reset behavior shared by live and offline MuJoCo apps.
  After alignment, the existing MuJoCo runtime still holds the previous command for a missing detection and advances
  exactly one command period per source frame.
- `retargeting_apps.pipelines.offline_retargeting` owns frame selection, progress reporting, robot-frame geometry,
  trajectory construction, and metadata construction. Artifact save and post actions remain in the outer app runner.
- Offline retargeting now loads only raw `stream_*` arrays through `load_offline_avp_trajectory`; stored
  `retarget_qpos` values are not materialized or used.

Before the split, three three-frame qpos baselines were captured from the promoted 760-frame fixture: default
simulation mode for frames `[0, 1, 2]`, smoothing-enabled real-world mode for `[0, 1, 2]`, and the selected window
`[5, 7, 9]`. After the split, all frame-index arrays matched and all three qpos arrays had maximum absolute
difference `0.0` from their pre-change values. The raw two-frame MuJoCo path still advances `0.1 s` without reading
stored robot qpos.

The Phase 4 verification on 2026-07-17 is 49 focused config/domain/import/ownership tests, 16 focused
artifact/replay tests, 11 focused MuJoCo backend and runtime tests, and 97 tests in the full headless suite. The full
suite completed in 12.61 s; `compileall`, isolated package imports, the core dependency audit, and
`git diff --check` passed. `ruff` and `black` remain unavailable in the environment, so those checks were not run.

## Reorganization Phase 5 Baseline

Phase 5 removes migration-only APIs and makes the final package graph executable documentation:

- Source, tests, docs, and ROS compatibility code use canonical package paths. Removed `retargeting.main`, app,
  pipeline, artifact, visualization, input/backend, and singular `teleoperation.input` paths have no import callers.
- Legacy `HandObservation` property aliases, unused `TeleoperationSession` methods, `RobotMujoco`, and the old
  `OfflineReplay/load_offline_replay` implementation are removed. ROS compatibility callers use canonical
  observation fields, raw-human loading, alignment services, and backend names.
- The three repeated path-or-mapping config loaders are replaced by the single
  `retargeting.config.io.load_config_source` implementation.
- The AST boundary contract now scans `retargeting`, `teleoperation`, `retargeting_apps`, and `retargeting_ros`,
  validates allowed cross-package edges, rejects removed module paths in source and ROS compatibility code, and
  proves the resulting directed package graph is acyclic.
- `AGENTS.md`, README, and the development guide describe final ownership, config domains,
  offline flows, and the `retargeting_apps.main:main` console composition root. `retargeting.core.*` remains the
  stable algorithm path and was not flattened.

The Phase 5 verification on 2026-07-17 is 49 focused config/domain/import/ownership tests, 7 optimizer/solver golden
tests, 16 artifact/replay tests, 11 MuJoCo backend and runtime tests, and 97 tests in the full headless suite. The
full suite completed in 15.85 s. `compileall` covered source, tests, and ROS compatibility Python files; isolated
optional-dependency imports, installed console dispatch, package discovery, and `git diff --check` passed. `ruff`
and `black` remain unavailable in the environment, so those checks were not run.

## Reorganization Phase 6 Final Verification

Phase 6 revalidated every final architecture and behavior contract without further source changes:

- The whole-package AST checks prove that `retargeting` has no outer-package, MuJoCo, AVP, OpenCV, Viser, or ROS
  imports; `teleoperation` has no `retargeting_apps` or `retargeting_ros` imports; and the declared four-package
  dependency graph is acyclic.
- Installed console metadata and the generated executable both resolve to `retargeting_apps.main:main`. An actual
  `retargeting app.id=__phase6_invalid__` invocation reached the new application registry and rejected the unknown
  id as expected. Package discovery includes `retargeting`, `teleoperation`, `retargeting_apps`, and
  `retargeting_ros`.
- Isolated package-root imports succeed while ROS, AVP, OpenCV, MediaPipe, MuJoCo, Open3D, and Viser are blocked.
- The Phase 4 pre-change fixture baseline was rerun for simulation frames `[0, 1, 2]`, real-world smoothing frames
  `[0, 1, 2]`, and window frames `[5, 7, 9]`. All indices matched, every qpos maximum absolute difference was `0.0`,
  and the five stored diagnostic arrays were present, frame-aligned, and finite.
- Focused MuJoCo tests continue to enforce one `0.05 s` command period per human frame, 25 physics steps at
  `0.002 s`, missing-detection hold, first-frame alignment, limiter, and overrun behavior.

The final verification on 2026-07-17 is 49 focused config/domain/import/ownership tests, 7 optimizer/solver golden
tests, 16 artifact/replay tests, 11 MuJoCo backend and runtime tests, and 97 tests in the full headless suite. The
full suite completed in 15.53 s. `compileall` covered source, tests, and ROS compatibility Python files;
installed-console dispatch, package discovery, isolated optional imports, and `git diff --check` passed. `ruff` and
`black` are not installed in the project environment, so those two checks were not run or installed.

## Passive mjviser Web Visualization Baseline

The optional offline MuJoCo Web viewer added on 2026-07-18 preserves the Phase 6 ownership and behavior contracts:

- `retargeting_apps.visualization.mjviser_live` owns the optional Viser server and passive `ViserMujocoScene`.
- `MujocoRobotBackend` remains headless and owns the only `MjModel`, `MjData`, and physics stepping path.
- The offline app publishes the initial backend state and one state after every processed or hold frame, optionally
  waits for the first browser, and stops the server on completion or failure.
- `loop=false` preserves one-pass playback. With `loop=true`, every repeated interval resets backend/session/limiter
  state, publishes the reset scene, and aligns again from the first selected valid source frame before continuing.
- The read-only `Joint angles` tab exposes all 23 actual Panda+Leap hinge values in radians. It follows compiled
  `jnt_qposadr` mappings, batches GUI updates atomically, and registers no state-mutating callbacks.
- `viewer.enabled=false` remains the default. Importing the application entrypoint, offline app, and viewer adapter
  succeeds while `mjviser`, Viser, MuJoCo, and the other optional adapters are blocked.
- Tests use fake server/scene objects and do not launch a Web server or browser. The installed `mjviser 0.0.14` API
  signatures were checked separately without constructing a server.

The latest 2026-07-18 verification is 19 focused MuJoCo/backend/offline/viewer tests and 107 tests in the full
headless suite. The focused suite completed in 3.81 s and the final full suite in 17.69 s. Hydra viewer/loop
composition, cross-layer reset behavior, `compileall`, optional-import boundaries, editable `mujoco-web` dependency
installation, and `git diff --check` also passed.

## MuJoCo Startup Move Policy

The current offline MuJoCo execution policy uses synchronized startup moves followed by direct targets:

- `simulator.startup_move_frames` defaults to `0` in the shared simulator config and is overridden to `1` by the
  offline app. Only successfully retargeted targets consume this count; hold frames do not.
- Startup waypoint counts use `ceil` and the configured per-joint speed limits, so every commanded increment is
  bounded and all joints reach the final actuator target on the same command period.
- After startup, requested targets are sent directly without an explicit speed limit. Actuator ctrlrange handling and
  MuJoCo force, servo, contact, and dynamics behavior remain active.
- Source-frame count is distinct from command-period count. Startup frames may advance multiple `0.05 s` periods;
  reset re-arms startup behavior, and mjviser observes every internal command period.
- Headless tests cover configuration validation, direct execution, strict startup waypoints, hold/reset counters,
  per-period callbacks, offline loop/viewer behavior, and a real Panda+Leap MJCF startup integration.

The latest 2026-07-18 verification is 42 focused config/MuJoCo/runtime/offline/viewer tests and 111 tests in the
complete headless suite. The focused suite completed in 4.98 s and the full suite in 17.73 s. A real one-frame
Panda+Leap run
planned 36 startup command periods and advanced `1.8 s` of simulation before consuming another source frame;
`compileall`, Hydra composition, and `git diff --check` also passed.

## Offline Session Composition Simplification

The offline artifact pipeline now composes `TeleoperationSession` and the shared AVP alignment helper
directly. The single-use `OfflineRetargetingService`, its intermediate `OfflineRetargetedFrame`, and its unused
configurable missing-detection policy have been removed. The pipeline still enables evaluation, aligns against the
configured initial robot qpos, skips missing observations, and copies qpos and diagnostics into replay frames.

The 2026-07-18 verification is 17 focused offline-pipeline/artifact/import tests and 111 tests in the complete
headless suite. The focused suite completed in 8.92 s and the full suite in 17.69 s. `compileall`, the 120-character
line-length check, direct package imports, removed-symbol searches, and `git diff --check` also passed.

## Unified Application Entry Contract

The application runner layout now has a one-to-one task contract:

- Every `configs/app/<id>.yaml` maps to `retargeting_apps.apps.<id>` and declares the same `app.id`.
- Every task module exposes `run(config, argv)` with no task-specific runner naming.
- `retargeting_apps.main` owns the explicit app whitelist, lazily imports only the selected task module, and contains
  no task-specific execution wrappers.
- `replay.py` owns the complete replay task entry. The old `viser_retargeting_visualize.py` name is removed.
- Shared `mujoco_runtime_builder.py` lives in `retargeting_apps.pipelines`, while `apps/` contains task entries only.
- Individual app modules are not standalone CLI modules. The installed `retargeting` command and
  `python -m retargeting_apps.main` remain the two public command forms.

The 2026-07-18 verification is 42 focused config/app/runtime/offline tests and 112 tests in the complete headless
suite. The focused suite completed in 6.92 s and the full suite in 17.74 s. `compileall`, the 120-character line-length
check, old-entrypoint residue search, and `git diff --check` also passed.

After moving the shared builder into `retargeting_apps.pipelines`, 23 focused app-ownership/runtime/offline tests
passed in 5.92 s and all 112 headless tests passed in 18.13 s. Direct import from the new canonical path, removal of
the old top-level path, `compileall`, the 120-character line-length check, and `git diff --check` also passed.

## Flat Teleoperation Architecture Final Verification

The 2026-07-18 flattening replaces the nested driver/runtime/session controller chain with two peer top-level flows:

- `ExecutionFlow` directly owns one `HandInput`, one `HandObservationMapper`, `Retargeter`, qpos output filtering,
  optional raw-result evaluation, command policy, atomic `RobotBackend.execute()`, timing, observers, and reset.
- `BatchRetargetFlow` uses the same sensor/mapping/retarget/filter path but has no robot backend, command timing, hold
  result, or execution-only nullable fields; later missing detections are skipped.
- `AvpOnlineInput` and `AvpOfflineInput` both call `inputs/avp/common.py`; importing archived AVP input does not load
  `avp_stream`. `SensorHandSample` distinguishes a missing detection from finite end-of-stream.
- `RetargetingHandObservation` is the only canonical core solver input. `HandInput` is owned by teleoperation.
- Live and archived MuJoCo apps both receive a complete flow from `retargeting_apps.composition` and call
  `flow.run()`. ROS RGB callback compatibility decodes a sample and calls `flow.step(sample)`.
- `TeleoperationSession`, the MuJoCo runtime/driver/result modules, the generic observation adapter, the standalone
  AVP alignment helper, and `retargeting_apps/pipelines/` are removed.

Three pre-change fixture baselines were recomputed from `HEAD` in an isolated `/tmp` archive and compared to the new
batch flow: simulation frames `[0, 1, 2]`, real-world smoothing frames `[0, 1, 2]`, and window frames `[5, 7, 9]`.
All frame indices matched and every qpos maximum absolute difference was `0.0`.

The final verification is 8 focused replay tests, 24 focused MuJoCo/flow/offline/viewer tests, and 117 tests in the
complete headless suite. The focused suites completed in 2.20 s and 3.75 s; the full suite completed in 17.80 s.
`compileall`, the 120-character changed-source check, package import boundaries, and `git diff --check` passed.
`ruff` and `black` are not installed in the project environment and were not run or installed.

## Ideal Kinematic Backend

The 2026-07-18 ideal backend adds `KinematicRobotBackend` as the dependency-free execution target for command-faithful
visualization:

- Every accepted command is realized atomically with `actual_qpos == command_qpos`; reset synchronizes both states.
- Joint ranges, startup speed limits, missing-frame hold, pacing, and observers remain owned by `ExecutionFlow` and
  `QposCommandLimiter`.
- The backend contains no Pinocchio, MuJoCo, ROS, camera, or viewer dependency. Visualization consumes immutable
  command results through flow observers.

The focused backend/runtime/import suite passed 39 tests in 4.94 s, and all 134 headless tests passed in 17.37 s.
`compileall`, canonical package/direct imports, the changed-source line-length check, and `git diff --check` passed.
