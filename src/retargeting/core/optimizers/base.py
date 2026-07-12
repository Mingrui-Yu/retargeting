"""Base protocol and shared solver wiring for retargeting objectives."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np

from retargeting.core.solvers import create_callback_solver
from retargeting.core.kinematics.adaptor import RobotAdaptor


def extract_solver_params(optimizer_params: Dict) -> Dict:
    """Extract solver settings from runtime optimizer parameters.

    Args:
        optimizer_params: Runtime optimizer parameters containing ``solver_params``.

    Returns:
        Backend-specific solver parameters.
    """
    solver_params = optimizer_params.get("solver_params")
    if not isinstance(solver_params, dict):
        raise ValueError("Runtime optimizer params must include solver_params.")
    return dict(solver_params)


class RetargetOptimizer:
    """Shared solver lifecycle and joint-limit handling for retargeting objectives."""

    retargeting_type = "BASE"

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        joint_limit_overrides: Optional[List[Dict]] = None,
        solver: str = "nlopt",
        solver_params: Optional[Dict] = None,
    ) -> None:
        """Initialize an objective with its numerical solver backend.

        Args:
            robot_adaptor: Kinematics adapter used by objective functions.
            joint_limit_overrides: Optional per-joint bound overrides in DOA order.
            solver: Numeric solver backend name.
            solver_params: Backend-specific solver parameters.

        Returns:
            None.
        """
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model
        self.solver = solver.lower()
        self.opt_dim = robot_adaptor.doa
        self.opt = create_callback_solver(solver, self.opt_dim)
        self.opt.configure({} if solver_params is None else solver_params)
        self.joint_limits = robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        self.set_joint_limit(self.joint_limits)

    def apply_joint_limit_overrides(self, joint_limit_overrides: List[Dict]) -> None:
        """Apply configured lower and upper joint-limit overrides.

        Args:
            joint_limit_overrides: Bound override mappings indexed in DOA order.

        Returns:
            None.
        """
        for override in joint_limit_overrides:
            indices = override["indices"]
            if override.get("lower") is not None:
                self.joint_limits[indices, 0] = override["lower"]
            if override.get("upper") is not None:
                self.joint_limits[indices, 1] = override["upper"]

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon: float = 1e-3) -> None:
        """Configure solver bounds from robot joint limits.

        Args:
            joint_limits: Lower and upper bounds with shape ``(n_doa, 2)``.
            epsilon: Small padding retained for existing solver behavior.

        Returns:
            None.
        """
        if joint_limits.shape != (self.opt_dim, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dim, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def retarget(
        self,
        ref_values: Dict[str, np.ndarray],
        arm_qpos: Optional[np.ndarray] = None,
        fixed_qpos_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Optimize a qpos vector for the supplied objective references.

        Args:
            ref_values: Reference values used by the selected objective.
            arm_qpos: Optional qpos values fixed during optimization.
            fixed_qpos_indices: Optional DOA indices for ``arm_qpos``.

        Returns:
            Optimized qpos in robot-adapter DOA order.
        """
        if arm_qpos is not None:
            fixed_qpos = np.asarray(arm_qpos, dtype=np.float32)
            if fixed_qpos_indices is None:
                fixed_qpos_indices = np.arange(len(fixed_qpos), dtype=int)
            fixed_qpos_indices = np.asarray(fixed_qpos_indices, dtype=int)
            if fixed_qpos_indices.shape != fixed_qpos.shape:
                raise ValueError(
                    "fixed_qpos_indices must have the same shape as arm_qpos, "
                    f"got {fixed_qpos_indices.shape} and {fixed_qpos.shape}."
                )
            ref_values["fixed_qpos"] = fixed_qpos
            ref_values["fixed_qpos_indices"] = fixed_qpos_indices
        self.opt.set_min_objective(self.get_objective_function(ref_values))
        x_init = ref_values["qpos_doa_last"]
        try:
            qpos_doa = self.opt.optimize(x_init)
        except ValueError as exc:
            print(exc)
            qpos_doa = x_init
        qpos_doa = np.clip(qpos_doa, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return np.array(qpos_doa, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        """Build the callback accepted by the selected numerical solver.

        Args:
            ref_values: Objective-specific references for one retargeting frame.

        Returns:
            Solver objective callback.
        """
        raise NotImplementedError
