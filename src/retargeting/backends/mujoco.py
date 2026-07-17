"""Headless MuJoCo robot backend for online joint-position execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from retargeting.config import MujocoSimulationConfig, load_mujoco_simulation_config

try:
    import mujoco
except ModuleNotFoundError as _MUJOCO_IMPORT_ERROR:
    mujoco = None
else:
    _MUJOCO_IMPORT_ERROR = None


def _require_mujoco() -> Any:
    """Return the optional MuJoCo module or raise an actionable error.

    Args:
        None.

    Returns:
        Imported MuJoCo Python module.
    """
    if mujoco is None:
        raise ModuleNotFoundError(
            "MuJoCo support is not installed. Install it with `pip install -e \".[mujoco]\"`."
        ) from _MUJOCO_IMPORT_ERROR
    return mujoco


class MujocoRobotBackend:
    """Execute robot joint-position commands in MuJoCo without a viewer."""

    def __init__(
        self,
        model_path: str | Path,
        joint_names: Sequence[str],
        initial_qpos: Sequence[float],
        config: MujocoSimulationConfig | dict[str, Any] | None = None,
    ) -> None:
        """Load a model and bind configured robot joints to position actuators.

        Args:
            model_path: MJCF file containing the robot and its position actuators.
            joint_names: Command order shared with the retargeting robot config.
            initial_qpos: Initial joint configuration in ``joint_names`` order.
            config: Online MuJoCo timing and command-range settings.

        Returns:
            None.
        """
        mj = _require_mujoco()
        self.config = load_mujoco_simulation_config(config)
        self.joint_names = tuple(str(name) for name in joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique.")

        self.model_path = Path(model_path).resolve()
        self.model = mj.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = self.config.physics_timestep
        self.data = mj.MjData(self.model)
        self._joint_ids, self._qpos_addresses, self._qvel_addresses = self._resolve_joint_indices()
        self._actuator_ids = self._resolve_actuator_indices()
        self._ctrl_limited = self.model.actuator_ctrllimited[self._actuator_ids].astype(bool)
        self._ctrlrange = self.model.actuator_ctrlrange[self._actuator_ids].copy()
        self.initial_qpos = self._validate_qpos(initial_qpos, "initial_qpos")
        self.target_joint_pos = self.initial_qpos.copy()
        self.reset(self.initial_qpos)

    @property
    def control_period(self) -> float:
        """Return simulated seconds advanced by one high-level step.

        Args:
            None.

        Returns:
            Command period in seconds.
        """
        return self.config.control_period

    @property
    def physics_steps_per_command(self) -> int:
        """Return physics substeps executed for each retargeting frame.

        Args:
            None.

        Returns:
            Integer MuJoCo substep count.
        """
        return self.config.physics_steps_per_command

    @property
    def joint_ctrlrange(self) -> tuple[np.ndarray, np.ndarray]:
        """Return actuator control bounds in retargeting joint order.

        Args:
            None.

        Returns:
            Lower and upper control-bound arrays. Unbounded actuators use infinities.
        """
        lower = np.full(len(self.joint_names), -np.inf, dtype=float)
        upper = np.full(len(self.joint_names), np.inf, dtype=float)
        lower[self._ctrl_limited] = self._ctrlrange[self._ctrl_limited, 0]
        upper[self._ctrl_limited] = self._ctrlrange[self._ctrl_limited, 1]
        return lower, upper

    def _resolve_joint_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve scalar joint state addresses by configured name.

        Args:
            None.

        Returns:
            Arrays of joint ids, qpos addresses, and qvel addresses.
        """
        mj = _require_mujoco()
        joint_ids = []
        for name in self.joint_names:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MJCF is missing configured joint {name!r}.")
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in {int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)}:
                raise ValueError(f"Configured joint {name!r} must be a scalar hinge or slide joint.")
            joint_ids.append(joint_id)
        joint_ids_array = np.asarray(joint_ids, dtype=int)
        return (
            joint_ids_array,
            self.model.jnt_qposadr[joint_ids_array].astype(int),
            self.model.jnt_dofadr[joint_ids_array].astype(int),
        )

    def _resolve_actuator_indices(self) -> np.ndarray:
        """Resolve exactly one joint-transmission actuator for every command joint.

        Args:
            None.

        Returns:
            Actuator ids in configured retargeting joint order.
        """
        mj = _require_mujoco()
        actuator_ids = []
        for name, joint_id in zip(self.joint_names, self._joint_ids):
            candidates = np.flatnonzero(
                (self.model.actuator_trntype == int(mj.mjtTrn.mjTRN_JOINT))
                & (self.model.actuator_trnid[:, 0] == joint_id)
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Configured joint {name!r} requires exactly one joint actuator, found {len(candidates)}."
                )
            actuator_id = int(candidates[0])
            gain = float(self.model.actuator_gainprm[actuator_id, 0])
            position_bias = float(self.model.actuator_biasprm[actuator_id, 1])
            gear = self.model.actuator_gear[actuator_id]
            if gain <= 0 or not np.isclose(position_bias, -gain) or not np.allclose(
                gear, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ):
                raise ValueError(
                    f"Actuator for joint {name!r} must be a unit-gear position servo whose ctrl is target qpos."
                )
            actuator_ids.append(actuator_id)
        return np.asarray(actuator_ids, dtype=int)

    def _validate_qpos(self, qpos: Sequence[float], field_name: str) -> np.ndarray:
        """Validate and copy a command vector in configured joint order.

        Args:
            qpos: Joint-position values to validate.
            field_name: Name used in validation errors.

        Returns:
            Finite one-dimensional float array.
        """
        values = np.asarray(qpos, dtype=float)
        expected_shape = (len(self.joint_names),)
        if values.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {expected_shape}, got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{field_name} must contain only finite values.")
        return values.copy()

    def _apply_ctrlrange(self, qpos: np.ndarray) -> np.ndarray:
        """Apply the configured actuator-range policy to a validated command.

        Args:
            qpos: Validated command in configured joint order.

        Returns:
            Command accepted by the actuator control ranges.
        """
        lower, upper = self.joint_ctrlrange
        outside = (qpos < lower) | (qpos > upper)
        if outside.any() and self.config.ctrlrange_policy == "error":
            details = [
                f"{self.joint_names[i]}={qpos[i]:.6g} not in [{lower[i]:.6g}, {upper[i]:.6g}]"
                for i in np.flatnonzero(outside)
            ]
            raise ValueError("Joint command exceeds MuJoCo actuator ctrlrange: " + ", ".join(details))
        return np.clip(qpos, lower, upper)

    def reset(self, qpos: Sequence[float] | None = None) -> None:
        """Reset all simulation state and synchronize position targets.

        Args:
            qpos: Optional reset configuration; defaults to configured initial qpos.

        Returns:
            None.
        """
        mj = _require_mujoco()
        reset_qpos = self.initial_qpos if qpos is None else self._validate_qpos(qpos, "qpos")
        reset_qpos = self._apply_ctrlrange(reset_qpos)
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = reset_qpos
        self.data.ctrl[self._actuator_ids] = reset_qpos
        self.target_joint_pos = reset_qpos.copy()
        mj.mj_forward(self.model, self.data)

    def ctrl_joint_pos(self, qpos: Sequence[float]) -> np.ndarray:
        """Set one position target without advancing simulation time.

        Args:
            qpos: Desired joint positions in configured retargeting order.

        Returns:
            Applied command after the configured actuator-range policy.
        """
        command = self._apply_ctrlrange(self._validate_qpos(qpos, "qpos"))
        self.data.ctrl[self._actuator_ids] = command
        self.target_joint_pos = command.copy()
        return command.copy()

    def step(self) -> None:
        """Advance one complete 20 Hz retargeting command period.

        Args:
            None.

        Returns:
            None.
        """
        mj = _require_mujoco()
        mj.mj_step(self.model, self.data, nstep=self.physics_steps_per_command)

    def get_joint_pos(self) -> np.ndarray:
        """Return simulated joint positions in retargeting order.

        Args:
            None.

        Returns:
            Current joint-position vector.
        """
        return self.data.qpos[self._qpos_addresses].copy()

    def get_joint_vel(self) -> np.ndarray:
        """Return simulated joint velocities in retargeting order.

        Args:
            None.

        Returns:
            Current joint-velocity vector.
        """
        return self.data.qvel[self._qvel_addresses].copy()

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the last applied position command.

        Args:
            None.

        Returns:
            Target joint-position vector.
        """
        return self.target_joint_pos.copy()

    def get_joint_torques(self) -> np.ndarray:
        """Return actuator forces in configured joint order.

        Args:
            None.

        Returns:
            Current actuator force vector.
        """
        return self.data.actuator_force[self._actuator_ids].copy()

    def get_diagnostics(self) -> dict[str, float]:
        """Return compact headless simulation diagnostics for the current state.

        Args:
            None.

        Returns:
            Scalar simulation time, tracking, velocity, force, and contact metrics.
        """
        tracking_error = self.get_joint_pos() - self.target_joint_pos
        qvel = self.get_joint_vel()
        force = self.get_joint_torques()
        return {
            "simulation_time": float(self.data.time),
            "tracking_error_max": float(np.max(np.abs(tracking_error))),
            "tracking_error_rms": float(np.sqrt(np.mean(np.square(tracking_error)))),
            "joint_velocity_max": float(np.max(np.abs(qvel))),
            "actuator_force_max": float(np.max(np.abs(force))),
            "contact_count": float(self.data.ncon),
        }


# Preserve the compatibility import used by the legacy ROS workspace.
RobotMujoco = MujocoRobotBackend


__all__ = ["MujocoRobotBackend", "RobotMujoco"]
