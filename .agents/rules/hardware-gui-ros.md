# Hardware, GUI, And ROS Rules

- Treat ROS, RViz, Realsense, MuJoCo viewer, Open3D GUI, cameras, Vision Pro live streaming, and real robot control as non-default actions.
- Do not launch them without explicit user instruction.
- Prefer offline fixtures and static parsers for tests.
- When CUDA is needed from Codex tools, use escalated execution; the default sandbox may hide `/dev/nvidia*` even when `nvidia-smi` works.
