from pathlib import Path

import pytest


FIXTURE = Path("tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz")


def _np():
    return pytest.importorskip("numpy")


def test_runtime_output_layout_defaults(tmp_path):
    from retargeting.benchmark_trajectory import resolve_benchmark_output_dirs, runtime_name_from_result
    from retargeting.offline_retarget import resolve_output_dir

    retarget_dir = resolve_output_dir(
        {"output_root": str(tmp_path / "outputs"), "run_name": "smoke"},
        robot_name="panda_leap_paxini",
        retargeting_type="VECTOR_WRIST_JOINT",
    )
    benchmark_dir, plot_dir = resolve_benchmark_output_dirs(
        {"output_root": str(tmp_path / "outputs"), "plot_root": str(tmp_path / "outputs")},
        retarget_dir,
    )

    assert retarget_dir == tmp_path / "outputs" / "smoke" / "retargeting"
    assert benchmark_dir == tmp_path / "outputs" / "smoke" / "benchmark"
    assert plot_dir == tmp_path / "outputs" / "smoke" / "plots"
    assert runtime_name_from_result(retarget_dir / "result.npz") == "smoke"


def test_offline_retargeting_result_artifact_round_trip(tmp_path):
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    pytest.importorskip("scipy")

    from retargeting.retargeting_replay import (
        create_robot_replay_context_from_metadata,
        run_offline_retargeting,
        trajectory_to_replay_frames,
    )
    from retargeting.trajectory_result import load_retargeting_trajectory, save_retargeting_trajectory

    context, trajectory, metadata = run_offline_retargeting(
        data_file=str(FIXTURE),
        robot_config_path="configs/robots/panda_leap_paxini.yaml",
        retargeting_config_path="configs/retargeting/vector_wrist_joint.yaml",
        start=0,
        end=1,
        stride=1,
    )
    output_dir = save_retargeting_trajectory(tmp_path / "result_run", trajectory, metadata)
    loaded_trajectory, loaded_metadata = load_retargeting_trajectory(output_dir)
    loaded_context = create_robot_replay_context_from_metadata(loaded_metadata)
    frames = trajectory_to_replay_frames(loaded_context, loaded_trajectory)

    assert (output_dir / "result.npz").is_file()
    assert (output_dir / "metadata.yaml").is_file()
    assert loaded_metadata.num_frames == 2
    assert loaded_metadata.qpos_dim == context.robot_adaptor.doa
    assert loaded_trajectory.retarget_qpos.shape == (2, context.robot_adaptor.doa)
    assert loaded_trajectory.hand_keypoints_world.shape == (2, 21, 3)
    assert "wrist" in loaded_trajectory.robot_frame_poses
    assert len(frames) == 2
    assert frames[0].qpos.shape == (context.robot_adaptor.doa,)
    assert np.isfinite(frames[0].hand_keypoints_world).all()


def test_benchmark_summary_from_saved_result(tmp_path):
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    pytest.importorskip("scipy")

    from retargeting.benchmark_trajectory import compute_benchmark_metrics, run_benchmark_from_config, summarize_metrics
    from retargeting.retargeting_replay import run_offline_retargeting
    from retargeting.trajectory_result import save_retargeting_trajectory

    _, trajectory, metadata = run_offline_retargeting(
        data_file=str(FIXTURE),
        robot_config_path="configs/robots/panda_leap_paxini.yaml",
        retargeting_config_path="configs/retargeting/vector_wrist_joint.yaml",
        start=0,
        end=1,
        stride=1,
    )
    result_dir = save_retargeting_trajectory(tmp_path / "result_run", trajectory, metadata)

    metrics = compute_benchmark_metrics(result_dir)
    summary = summarize_metrics(metrics)
    benchmark_output_dir, plot_output_dir = run_benchmark_from_config(
        {
            "result": str(result_dir),
            "output_dir": str(tmp_path / "benchmark"),
            "plot": False,
        }
    )

    assert plot_output_dir is None
    for metric_name in [
        "position_error",
        "orientation_error",
        "relative_position_error",
        "relative_position_to_wrist_error",
    ]:
        assert metrics[metric_name].shape[0] == 2
        assert np.isfinite(metrics[metric_name]).all()
        assert summary[metric_name]["count"] > 0
    assert metrics["optimization_time"].shape[0] == 1
    assert np.isfinite(metrics["optimization_time"]).all()
    assert summary["optimization_time"]["count"] > 0

    assert (benchmark_output_dir / "metrics.json").is_file()
    assert (benchmark_output_dir / "summary.csv").is_file()


def test_optimization_time_metric_skips_first_retargeted_frame():
    np = _np()

    from retargeting.benchmark_trajectory import optimization_time_from_errors

    optimization_time = optimization_time_from_errors(
        {"optimization_time": np.asarray([10.0, 0.1, 0.2])},
        n_frames=3,
    )

    np.testing.assert_allclose(optimization_time, np.asarray([0.1, 0.2]))


def test_benchmark_bar_summary_plot_from_metric_arrays(tmp_path):
    np = _np()
    pytest.importorskip("matplotlib")

    from retargeting.benchmark_trajectory import write_metric_plots

    metrics = {
        "position_error": np.asarray([[0.01, 0.02, 0.03, 0.04]]),
        "orientation_error": np.asarray([[0.1, 0.2, 0.3, 0.4]]),
        "relative_position_error": np.asarray([[0.01, 0.02, 0.03]]),
        "relative_position_to_wrist_error": np.asarray([[0.01, 0.02, 0.03, 0.04]]),
        "optimization_time": np.asarray([0.012, 0.014]),
    }
    write_metric_plots(metrics, tmp_path, result_label="smoke")

    assert (tmp_path / "benchmark_metric_means.png").is_file()
    assert (tmp_path / "benchmark_metric_means.pdf").is_file()
