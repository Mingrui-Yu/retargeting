from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from retargeting.config import resolve_project_path, to_plain_config_data
from retargeting.pipelines.offline_retargeting import create_robot_replay_context_from_metadata
from retargeting.artifacts.trajectory import load_retargeting_trajectory, resolve_result_paths
from retargeting.evaluation.robot_metrics import RobotBenchmark


_METRIC_PLOT_SPECS = (
    ("position_error", "Fingertip Global Position", "Error (cm)", 100.0),
    ("relative_position_to_wrist_error", "Fingertip Relative Position to Wrist", "Error (cm)", 100.0),
    ("relative_position_error", "Fingertip Relative Position to Thumb", "Error (cm)", 100.0),
    ("orientation_error", "Fingertip Orientation", "Error (rad)", 1.0),
    ("optimization_time", "Optimization Time", "Time (ms)", 1000.0),
)

def optimization_time_from_errors(errors: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    """Extract per-frame optimization time from saved trajectory errors.

    Args:
        errors: Error arrays loaded from result.npz.
        n_frames: Expected number of retargeted frames in the artifact.

    Returns:
        One-dimensional per-frame optimization time array in seconds, excluding the
        first retargeted frame as optimizer warm-up.
    """
    if "optimization_time" not in errors:
        raise KeyError("Saved result does not contain optimization_time.")
    values = np.asarray(errors["optimization_time"], dtype=float).reshape(-1)
    if values.shape[0] != n_frames:
        raise ValueError("optimization_time must contain one time value per retargeted frame.")
    if n_frames < 2:
        raise ValueError("At least two retargeted frames are required to skip the warm-up optimization time.")
    return values[1:]


def compute_benchmark_metrics(result_dir_or_file: str | Path) -> dict[str, np.ndarray]:
    """Compute benchmark metric arrays for a saved retargeting result.

    Args:
        result_dir_or_file: Result directory or direct result.npz path.

    Returns:
        Mapping from metric name to per-frame metric arrays.
    """
    trajectory, metadata = load_retargeting_trajectory(result_dir_or_file)
    context = create_robot_replay_context_from_metadata(metadata)
    benchmark = RobotBenchmark(
        robot_adaptor=context.robot_adaptor,
        benchmark_config=context.benchmark_config,
    )

    metrics: dict[str, list[np.ndarray]] = {
        "position_error": [],
        "orientation_error": [],
        "relative_position_error": [],
        "relative_position_to_wrist_error": [],
    }
    for qpos, hand_keypoints_world in zip(trajectory.retarget_qpos, trajectory.hand_keypoints_world):
        qpos_dof = context.robot_adaptor.forward_qpos(qpos)
        context.robot_model.compute_forward_kinematics(qpos_dof)
        metrics["position_error"].append(benchmark.position_error(qpos, hand_keypoints_world, 1))
        metrics["orientation_error"].append(benchmark.orientation_error(qpos, hand_keypoints_world, 1))
        metrics["relative_position_error"].append(benchmark.relative_position_error(qpos, hand_keypoints_world, 1))
        metrics["relative_position_to_wrist_error"].append(
            benchmark.relative_position_to_wrist_error(qpos, hand_keypoints_world, 1)
        )

    metric_arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    metric_arrays["optimization_time"] = optimization_time_from_errors(trajectory.errors, trajectory.n_frames)
    return metric_arrays


def summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize benchmark metric arrays.

    Args:
        metrics: Mapping from metric name to per-frame arrays.

    Returns:
        JSON-serializable summary with aggregate and per-component statistics.
    """
    summary: dict[str, Any] = {}
    for name, values in metrics.items():
        values = np.asarray(values, dtype=float)
        flat_values = values.reshape(-1)
        components = []
        if values.ndim > 1:
            for component_idx in range(values.shape[1]):
                component_values = values[:, component_idx]
                components.append(
                    {
                        "index": component_idx,
                        "mean": float(np.mean(component_values)),
                        "std": float(np.std(component_values)),
                        "min": float(np.min(component_values)),
                        "max": float(np.max(component_values)),
                    }
                )
        summary[name] = {
            "count": int(flat_values.size),
            "mean": float(np.mean(flat_values)),
            "std": float(np.std(flat_values)),
            "min": float(np.min(flat_values)),
            "max": float(np.max(flat_values)),
            "components": components,
        }
    return summary


def runtime_name_from_result(result_dir_or_file: str | Path) -> str:
    """Infer the runtime name from a saved retargeting result path.

    Args:
        result_dir_or_file: Result directory or direct result.npz path.

    Returns:
        Runtime directory name used to group retargeting, benchmark, and plot outputs.
    """
    result_dir, _, _ = resolve_result_paths(result_dir_or_file)
    if result_dir.name == "retargeting":
        return result_dir.parent.name
    return result_dir.name


def resolve_benchmark_output_dirs(config_data: dict[str, Any], result_dir_or_file: str | Path) -> tuple[Path, Path]:
    """Resolve benchmark and plot output directories.

    Args:
        config_data: Plain composed benchmark config.
        result_dir_or_file: Result directory or result.npz path used to derive default names.

    Returns:
        Tuple of benchmark output directory and plot output directory.
    """
    run_name = runtime_name_from_result(result_dir_or_file)

    benchmark_output_dir = config_data.get("output_dir")
    if benchmark_output_dir is None:
        benchmark_output_dir = resolve_project_path(str(config_data.get("output_root", "outputs"))) / run_name / "benchmark"
    else:
        benchmark_output_dir = resolve_project_path(str(benchmark_output_dir))

    plot_output_dir = config_data.get("plot_dir")
    if plot_output_dir is None:
        plot_output_dir = resolve_project_path(str(config_data.get("plot_root", "outputs"))) / run_name / "plots"
    else:
        plot_output_dir = resolve_project_path(str(plot_output_dir))
    return benchmark_output_dir, plot_output_dir


def write_summary_outputs(summary: dict[str, Any], output_dir: str | Path) -> None:
    """Write benchmark summary outputs.

    Args:
        summary: JSON-serializable benchmark summary.
        output_dir: Directory where metrics.json and summary.csv should be written.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "count", "mean", "std", "min", "max"])
        writer.writeheader()
        for metric_name, metric_summary in summary.items():
            writer.writerow(
                {
                    "metric": metric_name,
                    "count": metric_summary["count"],
                    "mean": metric_summary["mean"],
                    "std": metric_summary["std"],
                    "min": metric_summary["min"],
                    "max": metric_summary["max"],
                }
            )


def metric_mean_for_plot(values: np.ndarray, scale: float) -> float:
    """Compute the scalar trajectory mean used by the bar-summary plot.

    Args:
        values: Per-frame metric values, optionally with per-fingertip components.
        scale: Unit conversion multiplier applied after averaging.

    Returns:
        Scalar mean value for one metric across the whole trajectory.
    """
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.mean(flat_values) * scale)


def write_metric_plots(metrics: dict[str, np.ndarray], output_dir: str | Path, result_label: str = "Result") -> None:
    """Write a 1x5 bar-summary plot for one method on one trajectory.

    Args:
        metrics: Mapping from metric name to per-frame arrays.
        output_dir: Directory where plot files should be written.
        result_label: X-axis label for the single result bar.

    Returns:
        None.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps, rcParams
    from matplotlib import pyplot as plt

    from retargeting.utils.utils_plot import plotHistogram

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match the legacy benchmark plot style while keeping the new output to one bar per subplot.
    rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axs = plt.subplots(1, len(_METRIC_PLOT_SPECS), figsize=(26, 4))
    bar_colors = [colormaps["Pastel1"](0)]
    bar_labels = ["Error"]
    for ax, (metric_name, title, ylabel, scale) in zip(axs, _METRIC_PLOT_SPECS, strict=True):
        if metric_name not in metrics:
            raise KeyError(f"Missing metric for plot: {metric_name}")
        mean_value = metric_mean_for_plot(metrics[metric_name], scale)
        plt.sca(ax)
        plotHistogram(
            data=np.asarray([[mean_value]], dtype=float),
            x_labels=[result_label],
            bar_labels=bar_labels,
            bar_colors=bar_colors,
            x_width=0.7,
            border_width=0.2,
        )
        ax.grid(axis="y")
        ax.set_title(title, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xlabel("Trajectory", fontsize=24)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_metric_means.png", dpi=600)
    fig.savefig(output_dir / "benchmark_metric_means.pdf", dpi=600)
    plt.close(fig)


def run_benchmark_from_config(config: Any) -> tuple[Path, Path | None]:
    """Run benchmark computation from a composed config.

    Args:
        config: Hydra/OmegaConf config object or equivalent plain dictionary.

    Returns:
        Tuple of benchmark output directory and optional plot output directory.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected benchmark config to be a mapping.")
    result = config_data.get("result")
    if result is None:
        raise ValueError("Benchmark config requires `result=...`.")

    metrics = compute_benchmark_metrics(result)
    summary = summarize_metrics(metrics)
    benchmark_output_dir, plot_output_dir = resolve_benchmark_output_dirs(config_data, result)
    write_summary_outputs(summary, benchmark_output_dir)
    if bool(config_data.get("plot", True)):
        write_metric_plots(metrics, plot_output_dir, result_label=runtime_name_from_result(result))
        return benchmark_output_dir, plot_output_dir
    return benchmark_output_dir, None
