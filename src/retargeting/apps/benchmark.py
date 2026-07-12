"""CLI entrypoint for generating benchmark reports from saved trajectories."""

from __future__ import annotations

import sys
from typing import Any

from retargeting.config import resolve_project_path
from retargeting.pipelines.benchmark_report import run_benchmark_from_config


def compose_hydra_benchmark_config(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose the benchmark app configuration with Hydra.

    Args:
        overrides: Hydra override strings supplied after the module name.

    Returns:
        Resolved plain dictionary containing benchmark settings.
    """
    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "hydra-core is required for the benchmark entrypoint. "
            "Install the project dependencies, for example with `pip install -e .`."
        ) from exc
    config_dir = resolve_project_path("configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="benchmark", overrides=list(overrides or []))
    return OmegaConf.to_container(config, resolve=True)


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark report command-line application.

    Args:
        argv: Optional command-line arguments after the module name.

    Returns:
        None.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    benchmark_output_dir, plot_output_dir = run_benchmark_from_config(compose_hydra_benchmark_config(argv))
    print(f"Saved benchmark summary to {benchmark_output_dir}")
    if plot_output_dir is not None:
        print(f"Saved benchmark plots to {plot_output_dir}")


if __name__ == "__main__":
    main()
