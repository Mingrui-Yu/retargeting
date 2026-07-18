# Development Rules

- Use the project Python environment specified in `general.md` for Python tests and scripts.
- Assume the default environment is a headless server without GUI, ROS, RViz, Realsense, cameras, CUDA devices, or real hardware.
- Prefer headless, offline, hardware-free tests using fixtures and static parsers.
- Do not launch ROS, RViz, MuJoCo/Open3D viewers, cameras, Vision Pro streams, CUDA workloads, or real robot control without explicit user instruction.
- Keep optional dependencies lazy. Report missing dependencies or use explicit test skips instead of changing core behavior.
- When an explicitly requested CUDA task cannot access `/dev/nvidia*` in the sandbox, use escalated execution.
