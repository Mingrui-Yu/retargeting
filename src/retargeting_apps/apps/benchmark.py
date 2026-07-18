"""Benchmark application runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retargeting_apps.benchmark_report import run_benchmark_from_config


def run(config: Any, argv: list[str]) -> tuple[Path, Path | None]:
    """Generate and report benchmark artifacts for a saved trajectory.

    Args:
        config: Composed benchmark application configuration.
        argv: Command-line overrides accepted for a uniform app-runner interface.

    Returns:
        Benchmark output directory and optional plot output directory.
    """
    del argv
    benchmark_output_dir, plot_output_dir = run_benchmark_from_config(config)
    print(f"Saved benchmark summary to {benchmark_output_dir}")
    if plot_output_dir is not None:
        print(f"Saved benchmark plots to {plot_output_dir}")
    return benchmark_output_dir, plot_output_dir
