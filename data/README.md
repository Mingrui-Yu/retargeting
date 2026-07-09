# Data Directory

This directory may contain local experiment data, historical teleoperation recordings, and benchmark outputs.

The repository's headless tests and default replay config use the promoted fixture in `tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz` instead of depending on files under `data/`.

New generated outputs should go under `outputs/`:

- `outputs/teleop/`
- `outputs/simulation/`
- `outputs/benchmark/`
- `outputs/plots/`

Do not rely on files in this directory for fresh-checkout smoke tests unless they are explicitly promoted to `tests/fixtures/`.
