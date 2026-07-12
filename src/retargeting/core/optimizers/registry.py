"""Configuration-facing registry for supported retargeting objectives."""

from retargeting.core.optimizers.vector_wrist_joint import (
    VectorWristJointOptimizer,
    VectorWristJointOptimizerV2,
)


OPTIMIZER_CLASSES = {
    "VectorWristJointOptimizer": VectorWristJointOptimizer,
    "VectorWristJointOptimizerV2": VectorWristJointOptimizerV2,
}


def get_optimizer_class(class_name: str):
    """Resolve a configured optimizer class name.

    Args:
        class_name: Optimizer class name from a retargeting method config.

    Returns:
        Optimizer class used to construct the configured objective.
    """
    try:
        return OPTIMIZER_CLASSES[class_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer class: {class_name}") from exc
