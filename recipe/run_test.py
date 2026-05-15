import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.detection.config import config_factory
from nuscenes.prediction.helper import (
    convert_global_coords_to_local,
    convert_local_coords_to_global,
)
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points


rotation = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
box = Box(
    center=[1.0, 2.0, 3.0],
    size=[2.0, 4.0, 6.0],
    orientation=rotation,
    name="vehicle.car",
)
assert box.corners().shape == (3, 8)

box.translate(np.array([1.0, -1.0, 0.5]))
assert np.allclose(box.center, [2.0, 1.0, 3.5])

points = np.array([[1.0, 2.0], [2.0, 4.0], [1.0, 2.0]])
projected = view_points(points, np.eye(3), normalize=True)
assert np.allclose(projected[:2, :], [[1.0, 1.0], [2.0, 2.0]])

matrix = transform_matrix(np.array([1.0, 2.0, 3.0]), Quaternion(), inverse=False)
assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])

coords = np.array([[2.0, 2.0], [3.0, 2.0]])
translation = (1.0, 2.0, 0.0)
quat = tuple(Quaternion(axis=[0, 0, 1], angle=0.0).elements)
local = convert_global_coords_to_local(coords, translation, quat)
roundtrip = convert_local_coords_to_global(local, translation, quat)
assert np.allclose(roundtrip, coords)

config = config_factory("detection_cvpr_2019")
assert "car" in config.class_names
assert config.dist_ths == [0.5, 1.0, 2.0, 4.0]
