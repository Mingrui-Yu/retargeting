# AGENTS.md

Project-level entry point for AI agents.

## Project

This repository is a human-to-robot dexterous-hand retargeting codebase for the paper "Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation".

Main areas:

- Core retargeting and kinematics code: `ws_ros2/src/retargeting_benchmark/src/`
- ROS2 packages and launch files: `ws_ros2/src/`
- Robot descriptions and assets: `ws_ros2/src/my_robot_description/`, `assets/`
- Tests: `tests/`

## Default Instructions

Always read:

- `.agents/rules/core.md`
- `.agents/rules/headless-testing.md`

Load additional context only when the task matches the routing table below.

## Context Loading

| Task matches | Read |
|---|---|
| Phase 0, regression tests, smoke tests, replay fixture | `.agents/rules/phase0.md`, `.agents/contents/phase0-smoke-tests.md` |
| Server/GPU/CUDA/conda/debugging runtime environment | `.agents/contents/runtime-environment.md` |
| Architecture planning or repo reorganization | `.agents/contents/architecture-context.md` |
| ROS, RViz, Realsense, real robot, MuJoCo/Open3D viewer | `.agents/rules/hardware-gui-ros.md` |

## Source Of Truth

- Hard constraints live in `.agents/rules/`.
- Optional operational context lives in `.agents/contents/`.
- Do not duplicate long context in this file.
