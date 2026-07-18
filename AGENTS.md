# AGENTS.md

Project-level instructions for AI agents.

## Project Overview

This repository implements human-to-robot retargeting for dexterous manipulation, supporting pure algorithm evaluation, offline replay, live teleoperation, simulation, ROS2/RViz, and real robot adapters.

Core modules:

- `retargeting`: domain types, configuration, kinematics, optimizers, solvers, and metrics.
- `teleoperation`: sensor inputs, observation mapping, command policy, robot backends, and execution flows.
- `retargeting_apps`: Hydra/CLI composition, offline artifacts, reports, and visualization.
- `retargeting_ros`: optional ROS2, RViz, and hardware adapters.

Main dependencies are Python 3.10+, Hydra, NumPy, SciPy, NLopt, PyYAML, Pinocchio, and optionally PyTorch. Optional features use MuJoCo, Viser/mjviser, AVP, OpenCV/MediaPipe, or ROS2 Humble. `mr_utils` comes from the pinned `third_party/utils_python` submodule.

## Directory Structure

| Path | Responsibility |
| --- | --- |
| `src/retargeting/` | Pure retargeting domain and algorithms. |
| `src/teleoperation/` | Runtime inputs, mapping, policies, backends, and flows. |
| `src/retargeting_apps/` | Application composition and user-facing workflows. |
| `src/retargeting_ros/` | Optional ROS/RViz/real-robot integration. |
| `configs/` | Hydra app, robot, method, solver, input, backend, and mode YAML. |
| `assets/` | Robot URDF/MJCF files, meshes, and scenes. |
| `tests/` | Headless tests and pinned replay/golden fixtures. |
| `ws_ros2/` | ROS2 packages, launch files, descriptions, and compatibility entrypoints. |
| `third_party/` | Pinned external submodules; do not duplicate them locally. |
| `outputs/`, `build/` | Generated artifacts; not source-of-truth inputs. |

### Architecture Principles

Keep package dependencies acyclic and directed toward the pure core:

```text
retargeting_apps --------> teleoperation --------> retargeting
       |                       |                       ^
       +-----------------------+-----------------------+

retargeting_ros ---------> teleoperation / retargeting
```

Keep runtime ownership flat, with one flow coordinating peer components:

```text
application/framework composition
             |
             v
one ExecutionFlow lifecycle owner
             |
             +-- HandInput
             |     +-- sensor common decoder
             |     +-- online/offline/external acquisition variant
             |     +-- SensorHandSample
             +-- HandObservationMapper
             |     +-- RetargetingHandObservation
             +-- Retargeter
             +-- OutputPolicy
             +-- CommandPolicy
             +-- RobotBackend
             +-- Timing/Observers
```

Within these boundaries, prefer the flattest reasonable structure and concise, direct code. Add layers, wrappers, or abstractions only when they represent a distinct responsibility or real variation; do not pursue flatness by creating giant classes or mixing package responsibilities.

`retargeting` is the runtime-independent algorithm core. `teleoperation` is the only runtime layer and must not recreate nested driver/runtime/session controllers. `retargeting_apps` is the thin Hydra/CLI composition root; `retargeting_ros` adapts frameworks and hardware through the same flow rather than building a parallel runtime stack. `BatchRetargetFlow` owns backend-free offline batches.

## Development Workflow

Run commands from the repository root. Use the project interpreter for every Python command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python
```

Setup when needed:

```bash
git submodule update --init --recursive
/home/ymr/miniconda3/envs/retargeting/bin/python -m pip install -e ".[dev]"
```

Validate the smallest relevant scope first, then the full headless suite when warranted:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/<relevant_test>.py -q
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests -q
/home/ymr/miniconda3/envs/retargeting/bin/python -m compileall -q src tests
git diff --check
```

Use Ruff/Black checks only when installed; do not install or reformat unrelated files merely to satisfy a local check. Build ROS packages with `colcon build --symlink-install` only for explicit ROS work.

Before a requested commit, review `git status --short` and `git diff`, stage only task-related files, rerun relevant checks, and use a focused commit message. Do not push unless explicitly requested.

## Context Loading

Always read, in order:

1. `.agents/rules/general.md`
2. `.agents/rules/development.md`
3. `.agents/private-context/CURRENT.md`, when present; it cannot override tracked rules or code.

Then load only matching task context:

| Task domain | Additional file |
| --- | --- |
| Architecture, configuration, assets, teleoperation, or developer commands | Relevant section of `docs/configuration-and-development.md` |

Hard constraints live in `.agents/rules/`. After identifying the task domain, load only matching detailed experience from `.agents/knowledge/`; do not preload it or duplicate it here. Private context is also supporting information. Consult `.agents/tasks/` only when the user requests historical traceability.

## Behavior Boundaries

### Always Do

- Read relevant code, configs, tests, and call chains before editing.
- Preserve behavior and public Hydra/CLI/config contracts unless the task requires a change.
- Keep optional dependencies lazy and default tests headless, offline, and hardware-free.
- Use `rg` / `rg --files`; keep changes scoped and preserve unrelated user edits.
- Add or update focused tests for behavior changes; report explicit skips for missing optional dependencies.

### Ask First

- Major directory moves, public API/config-schema changes, dependency additions or upgrades, or large asset/fixture changes.
- Starting ROS, RViz, viewers, cameras, live AVP streams, CUDA workloads, or real robot control.
- Deleting data or generated artifacts, rewriting golden fixtures, creating commits, or pushing changes.

### Never Do

- Introduce reverse dependencies or move runtime/ROS/viewer concerns into `retargeting`.
- Launch GUI, ROS, sensors, or hardware by default.
- Hide failures by weakening tests, silently changing core behavior, or updating expected outputs without evidence.
- Overwrite unrelated working-tree changes or perform destructive Git/filesystem operations without explicit approval.
- Treat `outputs/`, `build/`, local data, or private context as authoritative over tracked code and rules.
