from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


CallbackObjective = Callable[[np.ndarray, np.ndarray], float]


class CallbackSolver(Protocol):
    """Shared interface for solvers that can consume the current numeric callback objective."""

    name: str

    def set_lower_bounds(self, lower_bounds: list[float]) -> None:
        """Set lower bounds for every optimization variable.

        Args:
            lower_bounds: Lower bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        ...

    def set_upper_bounds(self, upper_bounds: list[float]) -> None:
        """Set upper bounds for every optimization variable.

        Args:
            upper_bounds: Upper bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        ...

    def configure(self, params: dict[str, Any]) -> None:
        """Configure backend-specific solver options.

        Args:
            params: Backend-specific solver parameters.

        Returns:
            None.
        """
        ...

    def set_min_objective(self, objective: CallbackObjective) -> None:
        """Attach the objective callback for the next solve.

        Args:
            objective: Callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        ...

    def optimize(self, x_init: np.ndarray) -> np.ndarray:
        """Run the backend solver from the initial state.

        Args:
            x_init: Initial optimization vector.

        Returns:
            Optimized vector returned by the backend.
        """
        ...


@dataclass
class _CallbackSolverState:
    """Mutable state shared by callback-style optimization problems."""

    opt_dim: int
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    objective: CallbackObjective | None = None


class _BaseCallbackSolver:
    """Base class for numeric callback solvers used by current retarget objectives."""

    name = "base"

    def __init__(self, opt_dim: int):
        """Initialize backend-independent solver state.

        Args:
            opt_dim: Number of optimization variables.

        Returns:
            None.
        """
        self._state = _CallbackSolverState(opt_dim=opt_dim)

    @property
    def opt_dim(self) -> int:
        """Return the optimization dimension.

        Args:
            None.

        Returns:
            Number of optimization variables.
        """
        return self._state.opt_dim

    def set_lower_bounds(self, lower_bounds: list[float]) -> None:
        """Set lower bounds for every optimization variable.

        Args:
            lower_bounds: Lower bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        self._state.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self._validate_bound_shape(self._state.lower_bounds, "lower_bounds")

    def set_upper_bounds(self, upper_bounds: list[float]) -> None:
        """Set upper bounds for every optimization variable.

        Args:
            upper_bounds: Upper bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        self._state.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self._validate_bound_shape(self._state.upper_bounds, "upper_bounds")

    def configure(self, params: dict[str, Any]) -> None:
        """Configure backend-specific solver options.

        Args:
            params: Backend-specific solver parameters.

        Returns:
            None.
        """
        _ = params

    def set_min_objective(self, objective: CallbackObjective) -> None:
        """Attach the objective callback for the next solve.

        Args:
            objective: Callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        self._state.objective = objective

    def _validate_ready(self) -> None:
        """Validate that the common state is ready for a solve.

        Args:
            None.

        Returns:
            None.
        """
        if self._state.objective is None:
            raise ValueError("Solver objective has not been set.")
        if self._state.lower_bounds is None or self._state.upper_bounds is None:
            raise ValueError("Solver bounds have not been set.")

    def _validate_bound_shape(self, bounds: np.ndarray, name: str) -> None:
        """Validate one bound vector against the optimization dimension.

        Args:
            bounds: Bound vector to validate.
            name: Bound name used in the error message.

        Returns:
            None.
        """
        expected_shape = (self.opt_dim,)
        if bounds.shape != expected_shape:
            raise ValueError(f"Expected {name} shape {expected_shape}, got {bounds.shape}.")


class NloptSlsqpSolver(_BaseCallbackSolver):
    """nlopt SLSQP adapter for the current callback objective API."""

    name = "nlopt_slsqp"

    def __init__(self, opt_dim: int):
        """Initialize one persistent nlopt optimizer instance.

        Args:
            opt_dim: Number of optimization variables.

        Returns:
            None.
        """
        super().__init__(opt_dim)

        import nlopt

        self._opt = nlopt.opt(nlopt.LD_SLSQP, self.opt_dim)

    def configure(self, params: dict[str, Any]) -> None:
        """Configure NLopt SLSQP-specific options.

        Args:
            params: Solver params. Supported keys include `ftol_abs` and `maxtime`.
                Non-positive `maxtime` disables the wall-clock time limit.

        Returns:
            None.
        """
        if "ftol_abs" in params:
            self._opt.set_ftol_abs(float(params["ftol_abs"]))
        if "maxtime" in params:
            maxtime = float(params["maxtime"])
            self._opt.set_maxtime(maxtime if maxtime > 0.0 else 0.0)

    def set_lower_bounds(self, lower_bounds: list[float]) -> None:
        """Set lower bounds on both the adapter state and the persistent nlopt optimizer.

        Args:
            lower_bounds: Lower bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        super().set_lower_bounds(lower_bounds)
        self._opt.set_lower_bounds(self._state.lower_bounds.tolist())

    def set_upper_bounds(self, upper_bounds: list[float]) -> None:
        """Set upper bounds on both the adapter state and the persistent nlopt optimizer.

        Args:
            upper_bounds: Upper bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        super().set_upper_bounds(upper_bounds)
        self._opt.set_upper_bounds(self._state.upper_bounds.tolist())

    def set_min_objective(self, objective: CallbackObjective) -> None:
        """Attach the objective callback to the persistent nlopt optimizer.

        Args:
            objective: Callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        super().set_min_objective(objective)
        self._opt.set_min_objective(self._state.objective)

    def optimize(self, x_init: np.ndarray) -> np.ndarray:
        """Run the persistent nlopt SLSQP optimizer from the initial state.

        Args:
            x_init: Initial optimization vector.

        Returns:
            Optimized vector returned by nlopt.
        """
        self._validate_ready()
        return self._opt.optimize(np.asarray(x_init, dtype=np.float64))


class _ScipySlsqpObjectiveBuilder:
    """Convert the callback objective into scipy SLSQP `fun` and `jac` callables."""

    def __init__(self, objective: CallbackObjective | None = None):
        """Store the backend-independent callback objective.

        Args:
            objective: Optional callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        self.objective = objective

    def set_objective(self, objective: CallbackObjective) -> None:
        """Replace the callback objective used by the reusable scipy adapter.

        Args:
            objective: Callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        self.objective = objective

    def _require_objective(self) -> CallbackObjective:
        """Return the current objective or fail before scipy evaluates the adapter.

        Args:
            None.

        Returns:
            Current callback objective.
        """
        if self.objective is None:
            raise ValueError("Scipy objective has not been set.")
        return self.objective

    def fun(self, x: np.ndarray) -> float:
        """Evaluate only the scalar objective value.

        Args:
            x: Optimization vector supplied by scipy.

        Returns:
            Scalar objective value.
        """
        grad = np.array([], dtype=np.float64)
        return float(self._require_objective()(np.asarray(x, dtype=np.float64), grad))

    def jac(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the objective gradient.

        Args:
            x: Optimization vector supplied by scipy.

        Returns:
            Gradient vector with the same shape as `x`.
        """
        grad = np.zeros_like(np.asarray(x, dtype=np.float64))
        self._require_objective()(np.asarray(x, dtype=np.float64), grad)
        return grad


class _ScipyTimeLimit(RuntimeError):
    """Internal exception used to stop scipy SLSQP near the configured time budget."""


class ScipySlsqpSolver(_BaseCallbackSolver):
    """scipy SLSQP adapter for the current callback objective API."""

    name = "scipy_slsqp"

    def __init__(self, opt_dim: int):
        """Initialize reusable scipy adapter state.

        Args:
            opt_dim: Number of optimization variables.

        Returns:
            None.
        """
        super().__init__(opt_dim)
        self._objective_builder = _ScipySlsqpObjectiveBuilder()
        self._bounds: list[tuple[float, float]] | None = None
        self._options: dict[str, bool | float] = {"disp": False}
        self._maxtime: float | None = None

    def configure(self, params: dict[str, Any]) -> None:
        """Configure SciPy SLSQP-specific options.

        Args:
            params: Solver params. Supported keys include `ftol`, `maxtime`, and `maxiter`.
                Non-positive `maxtime` disables the callback time limit.

        Returns:
            None.
        """
        if "ftol" in params:
            self._options["ftol"] = float(params["ftol"])
        if "maxiter" in params:
            self._options["maxiter"] = int(params["maxiter"])
        if "maxtime" in params:
            maxtime = float(params["maxtime"])
            self._maxtime = maxtime if maxtime > 0.0 else None

    def set_lower_bounds(self, lower_bounds: list[float]) -> None:
        """Set lower bounds and invalidate cached scipy bounds.

        Args:
            lower_bounds: Lower bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        super().set_lower_bounds(lower_bounds)
        self._bounds = None

    def set_upper_bounds(self, upper_bounds: list[float]) -> None:
        """Set upper bounds and invalidate cached scipy bounds.

        Args:
            upper_bounds: Upper bound values with length equal to the optimization dimension.

        Returns:
            None.
        """
        super().set_upper_bounds(upper_bounds)
        self._bounds = None

    def set_min_objective(self, objective: CallbackObjective) -> None:
        """Attach the objective callback to the reusable scipy objective adapter.

        Args:
            objective: Callback with the nlopt-style signature `(x, grad) -> cost`.

        Returns:
            None.
        """
        super().set_min_objective(objective)
        self._objective_builder.set_objective(objective)

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Return cached scipy bounds, constructing them after bound changes.

        Args:
            None.

        Returns:
            Bounds in scipy's `(lower, upper)` sequence format.
        """
        if self._bounds is None:
            self._bounds = list(zip(self._state.lower_bounds.tolist(), self._state.upper_bounds.tolist()))
        return self._bounds

    def _get_options(self) -> dict[str, bool | float]:
        """Return scipy options for the next minimize call.

        Args:
            None.

        Returns:
            Copy of the configured scipy minimize options.
        """
        return dict(self._options)

    def optimize(self, x_init: np.ndarray) -> np.ndarray:
        """Run scipy SLSQP from the initial state.

        Args:
            x_init: Initial optimization vector.

        Returns:
            Optimized vector returned by scipy, or the latest iterate if the time budget is reached.
        """
        self._validate_ready()

        from scipy.optimize import minimize

        x_latest = np.asarray(x_init, dtype=np.float64).copy()
        start_time = time.monotonic()

        def callback(xk: np.ndarray) -> None:
            """Track the latest iterate and stop when the time budget is exceeded.

            Args:
                xk: Current scipy iterate.

            Returns:
                None.
            """
            nonlocal x_latest
            x_latest = np.asarray(xk, dtype=np.float64).copy()
            if self._maxtime is not None and time.monotonic() - start_time > self._maxtime:
                raise _ScipyTimeLimit()

        try:
            result = minimize(
                self._objective_builder.fun,
                x_latest,
                jac=self._objective_builder.jac,
                bounds=self._get_bounds(),
                method="SLSQP",
                callback=callback,
                options=self._get_options(),
            )
        except _ScipyTimeLimit:
            return x_latest

        if result.x is None or not np.isfinite(result.x).all():
            return np.asarray(x_init, dtype=np.float64)
        return np.asarray(result.x, dtype=np.float64)


def create_callback_solver(name: str, opt_dim: int) -> CallbackSolver:
    """Create a numeric callback solver by name.

    Args:
        name: Solver backend name. Supported values are `nlopt`, `nlopt_slsqp`, `scipy`, and `scipy_slsqp`.
        opt_dim: Number of optimization variables.

    Returns:
        Solver adapter implementing the current callback objective API.
    """
    normalized_name = name.lower()
    if normalized_name in {"nlopt", "nlopt_slsqp"}:
        return NloptSlsqpSolver(opt_dim)
    if normalized_name in {"scipy", "scipy_slsqp"}:
        return ScipySlsqpSolver(opt_dim)
    raise ValueError(f"Unsupported callback solver: {name}")
