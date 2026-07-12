"""Compatibility checks for the vendored ``mr_utils`` package."""

from __future__ import annotations

import numpy as np

from mr_utils import utils_calc, utils_mano, utils_torch


def test_vendored_utils_provide_retargeting_interfaces() -> None:
    """Verify the external package exposes every interface used by retargeting.

    Args:
        None.

    Returns:
        None.
    """
    assert callable(utils_calc.batchPosRotVec2Isometry3d)
    assert callable(utils_calc.posRotMat2Isometry3d)
    assert callable(utils_calc.transformPositions)
    assert callable(utils_torch.quaternion_angular_error)
    assert utils_mano.MANO_FINGERTIP_INDEX == [4, 8, 12, 16, 20]


def test_vendored_utils_preserve_required_transform_behavior() -> None:
    """Verify the utility transforms retain the behavior used by core code.

    Args:
        None.

    Returns:
        None.
    """
    pose = utils_calc.posRotMat2Isometry3d([1.0, 2.0, 3.0], np.eye(3))
    transformed = utils_calc.transformPositions([[0.0, 0.0, 0.0]], target_frame_pose=pose)

    # ``transformPositions`` maps world coordinates into the supplied frame.
    np.testing.assert_allclose(transformed, [[-1.0, -2.0, -3.0]])
