import sys
import types

import numpy as np


def test_retarget_injects_explicit_fixed_qpos_indices(monkeypatch):
    """Verify fixed qpos injection is independent from hand-specific index assumptions.

    Args:
        monkeypatch: Pytest fixture used to replace the solver factory.

    Returns:
        None.
    """

    class FakeSolver:
        """Solver fake that evaluates the active callback once."""

        def __init__(self):
            """Initialize mutable solver state.

            Args:
                None.

            Returns:
                None.
            """
            self.objective = None

        def configure(self, params):
            """Accept backend params without using them.

            Args:
                params: Backend-specific params.

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
            """Store the active objective callback.

            Args:
                objective: Objective callback.

            Returns:
                None.
            """
            self.objective = objective

        def optimize(self, x_init):
            """Evaluate the objective once and return the initial vector.

            Args:
                x_init: Initial optimization vector.

            Returns:
                Copy of the initial vector.
            """
            self.objective(x_init, np.zeros_like(x_init))
            return x_init.copy()

    class FakeRobotModel:
        """Robot model fake with only joint limits."""

        joint_limits = np.asarray([[-1.0, 1.0]] * 4, dtype=float)

    class FakeRobotAdaptor:
        """Robot adaptor fake with identity qpos mapping."""

        robot_model = FakeRobotModel()
        doa = 4

        def backward_qpos(self, qpos):
            """Return qpos unchanged.

            Args:
                qpos: Joint position vector.

            Returns:
                Copy of the input vector.
            """
            return np.asarray(qpos, dtype=float).copy()

    from retargeting.core.optimizers import base as optimizer_base
    from retargeting.core.optimizers import vector_wrist_joint as retarget_optimizer

    monkeypatch.setattr(optimizer_base, "create_callback_solver", lambda _solver, _opt_dim: FakeSolver())

    class RecordingOptimizer(retarget_optimizer.RetargetOptimizer):
        """Minimal optimizer that records fixed qpos values from ref_values."""

        def __init__(self):
            """Initialize the recording optimizer.

            Args:
                None.

            Returns:
                None.
            """
            super().__init__(FakeRobotAdaptor(), solver_params={})
            self.recorded_fixed_qpos = None
            self.recorded_fixed_qpos_indices = None

        def get_objective_function(self, ref_values):
            """Build a callback that records fixed qpos metadata.

            Args:
                ref_values: Retargeting reference values.

            Returns:
                Objective callback.
            """

            def objective(_x, _grad):
                """Record fixed qpos metadata and return zero cost.

                Args:
                    _x: Optimization vector.
                    _grad: Gradient buffer.

                Returns:
                    Scalar objective value.
                """
                self.recorded_fixed_qpos = ref_values["fixed_qpos"].copy()
                self.recorded_fixed_qpos_indices = ref_values["fixed_qpos_indices"].copy()
                return 0.0

            return objective

    optimizer = RecordingOptimizer()
    ref_values = {"qpos_doa_last": np.zeros(4)}

    optimizer.retarget(
        ref_values,
        arm_qpos=np.asarray([0.25, -0.5], dtype=float),
        fixed_qpos_indices=np.asarray([1, 3], dtype=int),
    )

    np.testing.assert_allclose(optimizer.recorded_fixed_qpos, [0.25, -0.5])
    np.testing.assert_array_equal(optimizer.recorded_fixed_qpos_indices, [1, 3])
    np.testing.assert_array_equal(ref_values["fixed_qpos_indices"], [1, 3])


def test_nlopt_solver_reuses_optimizer_between_objective_updates(monkeypatch):
    """Verify the nlopt adapter creates one backend optimizer and only replaces objectives.

    Args:
        monkeypatch: Pytest fixture used to install a fake nlopt module.

    Returns:
        None.
    """

    class FakeNloptOpt:
        """Small fake for the subset of nlopt.opt used by the adapter."""

        def __init__(self, algorithm, opt_dim):
            """Store constructor arguments and mutable optimizer settings.

            Args:
                algorithm: nlopt algorithm identifier supplied by the adapter.
                opt_dim: Number of optimization variables.

            Returns:
                None.
            """
            self.algorithm = algorithm
            self.opt_dim = opt_dim
            self.lower_bounds = None
            self.upper_bounds = None
            self.ftol_abs = None
            self.maxtime = None
            self.objective = None

        def set_lower_bounds(self, lower_bounds):
            """Record lower bounds passed to the fake backend.

            Args:
                lower_bounds: Lower bound values.

            Returns:
                None.
            """
            self.lower_bounds = lower_bounds

        def set_upper_bounds(self, upper_bounds):
            """Record upper bounds passed to the fake backend.

            Args:
                upper_bounds: Upper bound values.

            Returns:
                None.
            """
            self.upper_bounds = upper_bounds

        def set_ftol_abs(self, ftol_abs):
            """Record the absolute objective tolerance.

            Args:
                ftol_abs: Absolute objective tolerance.

            Returns:
                None.
            """
            self.ftol_abs = ftol_abs

        def set_maxtime(self, maxtime):
            """Record the per-solve time budget.

            Args:
                maxtime: Time budget in seconds.

            Returns:
                None.
            """
            self.maxtime = maxtime

        def set_min_objective(self, objective):
            """Record the active objective callback.

            Args:
                objective: Objective callback supplied by the adapter.

            Returns:
                None.
            """
            self.objective = objective

        def optimize(self, x_init):
            """Evaluate the active objective once and return the initial vector.

            Args:
                x_init: Initial optimization vector.

            Returns:
                Copy of the initial vector.
            """
            grad = np.zeros_like(x_init)
            self.objective(x_init, grad)
            return x_init.copy()

    created_opts = []

    def create_fake_opt(algorithm, opt_dim):
        """Create and record one fake nlopt optimizer.

        Args:
            algorithm: nlopt algorithm identifier supplied by the adapter.
            opt_dim: Number of optimization variables.

        Returns:
            Fake nlopt optimizer instance.
        """
        fake_opt = FakeNloptOpt(algorithm, opt_dim)
        created_opts.append(fake_opt)
        return fake_opt

    fake_nlopt = types.SimpleNamespace(LD_SLSQP=7, opt=create_fake_opt)
    monkeypatch.setitem(sys.modules, "nlopt", fake_nlopt)

    from retargeting.core.solvers import NloptSlsqpSolver

    objective_calls = []

    def objective_one(x, grad):
        """Record that the first objective was evaluated.

        Args:
            x: Optimization vector.
            grad: Gradient buffer.

        Returns:
            Scalar objective value.
        """
        objective_calls.append(("one", x.copy()))
        grad[:] = 1.0
        return 1.0

    def objective_two(x, grad):
        """Record that the second objective was evaluated.

        Args:
            x: Optimization vector.
            grad: Gradient buffer.

        Returns:
            Scalar objective value.
        """
        objective_calls.append(("two", x.copy()))
        grad[:] = 2.0
        return 2.0

    solver = NloptSlsqpSolver(2)
    solver.set_lower_bounds([-1.0, -2.0])
    solver.set_upper_bounds([1.0, 2.0])
    solver.configure({"ftol_abs": 1e-5, "maxtime": 0.01})

    solver.set_min_objective(objective_one)
    np.testing.assert_allclose(solver.optimize(np.array([0.0, 0.5])), [0.0, 0.5])

    solver.set_min_objective(objective_two)
    np.testing.assert_allclose(solver.optimize(np.array([0.25, 0.75])), [0.25, 0.75])

    assert len(created_opts) == 1
    fake_opt = created_opts[0]
    assert fake_opt.algorithm == fake_nlopt.LD_SLSQP
    assert fake_opt.opt_dim == 2
    assert fake_opt.lower_bounds == [-1.0, -2.0]
    assert fake_opt.upper_bounds == [1.0, 2.0]
    assert fake_opt.ftol_abs == 1e-5
    assert fake_opt.maxtime == 0.01
    assert fake_opt.objective is objective_two
    assert [call[0] for call in objective_calls] == ["one", "two"]


def test_nlopt_solver_non_positive_maxtime_disables_time_limit(monkeypatch):
    """Verify negative maxtime is mapped to NLopt's disabled time limit.

    Args:
        monkeypatch: Pytest fixture used to install a fake nlopt module.

    Returns:
        None.
    """

    class FakeNloptOpt:
        """Small fake that records the configured maxtime value."""

        def __init__(self, _algorithm, _opt_dim):
            """Initialize recorded maxtime.

            Args:
                _algorithm: Unused nlopt algorithm identifier.
                _opt_dim: Unused optimization dimension.

            Returns:
                None.
            """
            self.maxtime = None

        def set_maxtime(self, maxtime):
            """Record maxtime passed by the adapter.

            Args:
                maxtime: Time budget value passed to nlopt.

            Returns:
                None.
            """
            self.maxtime = maxtime

    fake_opt = FakeNloptOpt(None, None)
    fake_nlopt = types.SimpleNamespace(LD_SLSQP=7, opt=lambda _algorithm, _opt_dim: fake_opt)
    monkeypatch.setitem(sys.modules, "nlopt", fake_nlopt)

    from retargeting.core.solvers import NloptSlsqpSolver

    solver = NloptSlsqpSolver(2)
    solver.configure({"maxtime": -1})

    assert fake_opt.maxtime == 0.0


def test_scipy_solver_non_positive_maxtime_disables_time_limit():
    """Verify negative maxtime clears scipy's callback time limit.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.core.solvers import ScipySlsqpSolver

    solver = ScipySlsqpSolver(2)
    solver.configure({"maxtime": 0.01})
    assert solver._maxtime == 0.01

    solver.configure({"maxtime": -1})
    assert solver._maxtime is None
