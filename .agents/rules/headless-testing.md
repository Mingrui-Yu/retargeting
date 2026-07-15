# Headless Testing Rules

- Default environment is a server and may not expose GUI, ROS, RViz, Realsense, cameras, or real hardware.
- Use the project Python environment specified in `core.md` for Python tests and scripts.
- Prefer headless, offline, no-hardware, no-ROS tests.
- Do not start RViz, Open3D GUI, MuJoCo viewer, cameras, real robots, or ROS launch unless the user explicitly asks.
- If optional dependencies are missing, use explicit skips or report the missing dependency. Do not hide the failure by changing core behavior.
