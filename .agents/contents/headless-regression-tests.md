# Headless Regression Test Context

Run tests from the repository root with the project Python environment. Default tests must not start ROS, RViz, cameras, Vision Pro live streaming, a real robot, Open3D GUI, or a MuJoCo viewer.

The focused offline regression file is `tests/test_replay_smoke.py`. It currently covers:

- the Phase 4 robot-asset layout, including removal of obsolete root-level URDF links;
- the promoted 760-frame AVP fixture `tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz` and its first recorded `retarget_qpos`;
- static `AvpDetector.detect()` parsing with a fake live AVP dependency;
- configured Panda+Leap URDF loading and `RobotAdaptor` qpos/Jacobian mappings;
- Panda+Leap profile lower-bound overrides reaching both the optimizer and its solver;
- one `VectorWristJointOptimizer.retarget()` call for each supported SLSQP backend: NLopt and SciPy.

Focused command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_replay_smoke.py -q
```

Full headless regression command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests -q
```

The baseline verified on 2026-07-13 is 8 focused tests and 59 tests in the full suite. Update this document when the intended test scope changes; optional dependencies should produce explicit skips rather than a behavior change.
