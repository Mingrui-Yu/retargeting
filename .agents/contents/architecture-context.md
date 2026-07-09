# Architecture Context

Recommended direction:

- Make retargeting a clean Python core library.
- Keep ROS, hardware, and visualization as adapters around the core.
- Move robot-specific joints, links, paths, limits, and initial states into config.
- Keep large experiment outputs outside source control; keep only tiny fixtures in `tests/fixtures/`.
- Prefer CLI/offline replay as the first user-facing quickstart path.

Detailed architecture notes are in:

- `temps/agents/architecture_reorg_recommendations_zh_20260709_051243.md`
- `temps/agents/architecture_reorg_recommendations_20260709_051243.md`
