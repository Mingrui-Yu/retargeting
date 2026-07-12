from pathlib import Path
import time

import pytest


GOLDEN = Path("tests/fixtures/vector_wrist_joint_objective_golden.npz")
REPLAY_FIXTURE = Path("tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz")


class _NoopSolver:
    """Small solver fake used only to construct optimizer instances."""

    def configure(self, params):
        """Accept backend params without using them.

        Args:
            params: Backend-specific solver parameters.

        Returns:
            None.
        """

    def set_lower_bounds(self, lower_bounds):
        """Accept lower bounds without using them.

        Args:
            lower_bounds: Lower bound values.

        Returns:
            None.
        """

    def set_upper_bounds(self, upper_bounds):
        """Accept upper bounds without using them.

        Args:
            upper_bounds: Upper bound values.

        Returns:
            None.
        """

    def set_min_objective(self, objective):
        """Accept an objective without using it.

        Args:
            objective: Objective callback.

        Returns:
            None.
        """

    def optimize(self, x_init):
        """Return the initial value unchanged.

        Args:
            x_init: Initial optimization vector.

        Returns:
            Copy of the initial optimization vector.
        """
        return x_init.copy()


def _ordered_targets(profile_config):
    """Return target link names in the optimizer constructor format.

    Args:
        profile_config: Loaded robot-method retargeting profile configuration.

    Returns:
        Mapping with origin link names, task link names, and wrist link name.
    """
    target_config = profile_config.target
    link_pairs = target_config.link_pairs
    return {
        "origin_links_name": [pair[0] for pair in link_pairs],
        "task_links_name": [pair[1] for pair in link_pairs],
        "wrist_link_name": target_config.wrist_link_name,
    }


def _target_link_vectors(robot_model, robot_adaptor, targets, qpos_doa):
    """Build deterministic link-vector references from a nearby robot pose.

    Args:
        robot_model: Robot kinematics model.
        robot_adaptor: Robot adaptor used to convert DOA qpos to model qpos.
        targets: Optimizer target mapping with ordered link pairs.
        qpos_doa: Robot qpos in adaptor DOA order.

    Returns:
        Link vectors with one row for each configured target link pair.
    """
    qpos_dof = robot_adaptor.forward_qpos(qpos_doa)
    robot_model.compute_forward_kinematics(qpos_dof)

    vectors = []
    for origin_name, task_name in zip(targets["origin_links_name"], targets["task_links_name"]):
        origin_pos = robot_model.get_frame_pose(origin_name)[:3, 3]
        task_pos = robot_model.get_frame_pose(task_name)[:3, 3]
        vectors.append(task_pos - origin_pos)
    return vectors


def build_vector_wrist_joint_regression_case(monkeypatch, optimizer_class_name="VectorWristJointOptimizer"):
    """Construct a deterministic objective regression case for VectorWristJointOptimizer.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to replace the solver factory.
        optimizer_class_name: Optimizer class name to instantiate from `retargeting.retarget_optimizer`.

    Returns:
        Tuple of numpy module, objective callback, probe qpos, and optimization dimension.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("pinocchio")
    pytest.importorskip("torch")

    from retargeting import retarget_optimizer
    from retargeting.config import load_retargeting_config, load_retargeting_profile_config, load_robot_config
    from retargeting.robot_adaptor import RobotAdaptor
    from retargeting.robot_pinocchio import RobotPinocchio

    monkeypatch.setattr(retarget_optimizer, "create_callback_solver", lambda _solver, _opt_dim: _NoopSolver())

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    profile_config = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    retargeting_config = load_retargeting_config(profile_config.method)
    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    robot_adaptor = RobotAdaptor(robot_model, actuated_joints_name=list(robot_config.actuated_joints))
    targets = _ordered_targets(profile_config)

    optimizer_class = getattr(retarget_optimizer, optimizer_class_name)
    optimizer = optimizer_class(
        robot_adaptor=robot_adaptor,
        targets=targets,
        params={**retargeting_config.optimizer_params, "solver_params": {}},
    )

    qpos_base = np.load(REPLAY_FIXTURE)["retarget_qpos"][0].astype(float)
    offset = np.linspace(-0.006, 0.006, robot_adaptor.doa)
    x_probe = np.clip(qpos_base + offset, optimizer.joint_limits[:, 0] + 1e-5, optimizer.joint_limits[:, 1] - 1e-5)

    link_vectors = np.asarray(_target_link_vectors(robot_model, robot_adaptor, targets, qpos_base), dtype=float)
    link_offsets = np.linspace(-0.003, 0.003, link_vectors.size).reshape(link_vectors.shape)

    ref_qpos = qpos_base + np.linspace(0.004, -0.004, robot_adaptor.doa)
    qpos_last = qpos_base + np.linspace(-0.002, 0.002, robot_adaptor.doa)
    wrist_quat = np.asarray([0.97, 0.12, -0.08, 0.19], dtype=float)
    wrist_quat = wrist_quat / np.linalg.norm(wrist_quat)

    ref_values = {
        "links_vec": link_vectors + link_offsets,
        "wrist_quat": wrist_quat,
        "qpos_doa": ref_qpos,
        "qpos_doa_last": qpos_last,
        "weights": {
            "links_vec": np.linspace(0.2, 1.6, link_vectors.shape[0]),
            "wrist_rot": 0.1,
            "joint_pos": np.linspace(0.0, 0.5, robot_adaptor.doa),
            "joint_vel": np.linspace(0.02, 0.12, robot_adaptor.doa),
        },
    }

    objective = optimizer.get_objective_function(ref_values)
    return np, objective, x_probe, robot_adaptor.doa


def _measure_objective_runtime(np, objective, x_probe, opt_dim, with_grad, iterations=10, warmup=2):
    """Measure objective callback runtime for a deterministic probe.

    Args:
        np: Imported numpy module.
        objective: Objective callback with the nlopt-style `(x, grad) -> cost` signature.
        x_probe: Fixed optimization vector used for every callback evaluation.
        opt_dim: Optimization vector dimension.
        with_grad: Whether to request analytic gradient output.
        iterations: Number of timed callback evaluations.
        warmup: Number of untimed callback evaluations before measurement.

    Returns:
        Mapping containing median/mean runtime, last cost, and optional last gradient.
    """

    def make_grad():
        """Create the gradient buffer shape used by the selected measurement mode.

        Args:
            None.

        Returns:
            Gradient buffer passed to the objective callback.
        """
        if with_grad:
            return np.zeros(opt_dim, dtype=np.float64)
        return np.asarray([], dtype=np.float64)

    last_cost = None
    last_grad = None
    for _ in range(warmup):
        grad = make_grad()
        last_cost = objective(x_probe.copy(), grad)
        last_grad = grad

    times = []
    for _ in range(iterations):
        grad = make_grad()
        start = time.perf_counter()
        last_cost = objective(x_probe.copy(), grad)
        times.append(time.perf_counter() - start)
        last_grad = grad

    times = np.asarray(times, dtype=float)
    return {
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
        "last_cost": float(last_cost),
        "last_grad": last_grad.copy(),
    }


def test_vector_wrist_joint_objective_cost_and_gradient_regression(monkeypatch):
    """Pin the current VectorWristJoint objective scalar cost and analytic gradient.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to replace the solver factory.

    Returns:
        None.
    """
    np, objective, x_probe, opt_dim = build_vector_wrist_joint_regression_case(monkeypatch)
    expected = np.load(GOLDEN)

    grad = np.zeros(opt_dim, dtype=np.float64)
    cost = objective(x_probe.copy(), grad)
    cost_only = objective(x_probe.copy(), np.asarray([], dtype=np.float64))

    np.testing.assert_allclose(cost, expected["cost"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cost_only, expected["cost"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(grad, expected["grad"], rtol=1e-6, atol=1e-8)


def test_vector_wrist_joint_v2_matches_v1_objective_cost_and_gradient(monkeypatch):
    """Verify V2 preserves V1 objective semantics for the deterministic probe.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to replace the solver factory.

    Returns:
        None.
    """
    np, objective_v1, x_probe, opt_dim = build_vector_wrist_joint_regression_case(
        monkeypatch, "VectorWristJointOptimizer"
    )
    _, objective_v2, x_probe_v2, opt_dim_v2 = build_vector_wrist_joint_regression_case(
        monkeypatch, "VectorWristJointOptimizerV2"
    )
    np.testing.assert_allclose(x_probe_v2, x_probe, rtol=0, atol=0)
    assert opt_dim_v2 == opt_dim

    grad_v1 = np.zeros(opt_dim, dtype=np.float64)
    grad_v2 = np.zeros(opt_dim, dtype=np.float64)
    cost_v1 = objective_v1(x_probe.copy(), grad_v1)
    cost_v2 = objective_v2(x_probe.copy(), grad_v2)
    cost_only_v2 = objective_v2(x_probe.copy(), np.asarray([], dtype=np.float64))

    np.testing.assert_allclose(cost_v2, cost_v1, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cost_only_v2, cost_v1, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(grad_v2, grad_v1, rtol=1e-6, atol=1e-8)


def test_vector_wrist_joint_v2_speed_smoke(monkeypatch):
    """Report V1/V2 objective runtime while keeping correctness assertions.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to replace the solver factory.

    Returns:
        None.
    """
    np, objective_v1, x_probe, opt_dim = build_vector_wrist_joint_regression_case(
        monkeypatch, "VectorWristJointOptimizer"
    )
    _, objective_v2, x_probe_v2, opt_dim_v2 = build_vector_wrist_joint_regression_case(
        monkeypatch, "VectorWristJointOptimizerV2"
    )
    np.testing.assert_allclose(x_probe_v2, x_probe, rtol=0, atol=0)
    assert opt_dim_v2 == opt_dim

    measurements = {}
    for mode_name, with_grad in [("cost_only", False), ("cost_gradient", True)]:
        measurements[("v1", mode_name)] = _measure_objective_runtime(
            np, objective_v1, x_probe, opt_dim, with_grad=with_grad
        )
        measurements[("v2", mode_name)] = _measure_objective_runtime(
            np, objective_v2, x_probe, opt_dim, with_grad=with_grad
        )

        v1 = measurements[("v1", mode_name)]
        v2 = measurements[("v2", mode_name)]
        assert np.isfinite(v1["median_s"]) and v1["median_s"] > 0.0
        assert np.isfinite(v2["median_s"]) and v2["median_s"] > 0.0
        np.testing.assert_allclose(v2["last_cost"], v1["last_cost"], rtol=1e-8, atol=1e-10)
        if with_grad:
            np.testing.assert_allclose(v2["last_grad"], v1["last_grad"], rtol=1e-6, atol=1e-8)

    cost_only_v1 = measurements[("v1", "cost_only")]
    cost_only_v2 = measurements[("v2", "cost_only")]
    cost_gradient_v1 = measurements[("v1", "cost_gradient")]
    cost_gradient_v2 = measurements[("v2", "cost_gradient")]
    print(
        "VectorWristJoint objective speed "
        f"cost-only median: V1={cost_only_v1['median_s'] * 1000:.3f}ms, "
        f"V2={cost_only_v2['median_s'] * 1000:.3f}ms; "
        f"cost+grad median: V1={cost_gradient_v1['median_s'] * 1000:.3f}ms, "
        f"V2={cost_gradient_v2['median_s'] * 1000:.3f}ms"
    )
