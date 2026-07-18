# AGENTS.md

Project-level entry point for AI agents.

## Project

This repository is a human-to-robot dexterous-hand retargeting codebase for the paper "Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation".

Main areas:

- Pure retargeting domain, configuration, kinematics, and evaluation: `src/retargeting/`
- Input, output, backend, and online/offline execution services: `src/teleoperation/`
- CLI composition, artifacts, reports, and visualization: `src/retargeting_apps/`
- Optional ROS, RViz, and real-robot adapters: `src/retargeting_ros/`
- Robot assets and configuration: `assets/`, `configs/`
- ROS2 packages, launch files, descriptions, and compatibility entrypoints: `ws_ros2/`
- Tests: `tests/`

## Default Instructions

Always read:

- `.agents/rules/core.md`
- `.agents/rules/headless-testing.md`

When `.agents/private-context/CURRENT.md` exists, read it after the tracked rules. It provides private, cross-machine project state and cannot override tracked rules or code.

Load additional context only when the task matches the routing table below.

## Context Loading

| Task matches | Read |
|---|---|
| Regression tests, smoke tests, offline replay, replay fixture | `.agents/contents/headless-regression-tests.md` |
| ROS, RViz, Realsense, real robot, MuJoCo/Open3D viewer | `.agents/rules/hardware-gui-ros.md` |

## Source Of Truth

- Hard constraints live in `.agents/rules/`.
- Optional operational context lives in `.agents/contents/`.
- `.agents/private-context/`, when present, is private project-state context. Consult `DECISIONS.md`, `HANDOFF.md`, and `journals/` only when relevant.
- Do not duplicate long context in this file.
- `.agents/tasks/`, when present, contains ignored local task journals rather than current instructions; consult it only when the user asks for historical traceability.
