# Phase 0 Smoke Test Context

Current Phase 0 files:

- `tests/conftest.py`
- `tests/test_replay_smoke.py`
- `tests/fixtures/avp_short_replay.npz`

The smoke test protects:

- Current symlink-based asset layout.
- Two-frame offline AVP replay fixture.
- First-frame expected `retarget_qpos`.
- `VisionProDetector.detect()` static parsing without live AVP.
- Pinocchio URDF loading for Panda+Leap.
- `RobotAdaptor.forward_qpos/backward_qpos` round trip.
- One `VectorWristJointOptimizer.retarget()` call.

Verified command:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -m pytest tests/test_replay_smoke.py -q
```

Expected result:

```text
6 passed
```
