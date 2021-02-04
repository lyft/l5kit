from typing import Optional, Tuple

import cv2
import numpy as np
from random import randint

from l5kit.geometry import transform_point, transform_points


PREDICTED_POINTS_COLOR = (0, 255, 255)
TARGET_POINTS_COLOR = (255, 0, 255)
REFERENCE_TRAJ_COLOR = (255, 255, 0)
#  Arrows represent position + orientation.
ARROW_LENGTH_IN_PIXELS = 2
ARROW_THICKNESS_IN_PIXELS = 1
ARROW_TIP_LENGTH_IN_PIXELS = 1.8


def draw_arrowed_line(on_image: np.ndarray, position: np.ndarray, yaw: float, rgb_color: Tuple[int, int, int]) -> None:
    """
    Draw a single arrowed line in an RGB image
    Args:
        on_image (np.ndarray): the RGB image to draw onto
        position (np.ndarray): the starting position of the arrow
        yaw (float): the arrow orientation
        rgb_color (Tuple[int, int, int]): the arrow color

    Returns: None

    """
    start_pixel = np.array(position[:2])

    rot = np.eye(3)
    rot[:-1] = cv2.getRotationMatrix2D((0, 0), np.degrees(-yaw), 1.0)  # minus here because of cv2 rotations convention
    end_pixel = start_pixel + transform_point(np.asarray([ARROW_LENGTH_IN_PIXELS, 0]), rot)

    cv2.arrowedLine(
        on_image,
        tuple(start_pixel.astype(np.int32)),
        tuple(end_pixel.astype(np.int32)),
        rgb_color,
        thickness=ARROW_THICKNESS_IN_PIXELS,
        tipLength=ARROW_TIP_LENGTH_IN_PIXELS,
    )


def draw_trajectory(
        on_image: np.ndarray,
        positions: np.ndarray,
        rgb_color: Tuple[int, int, int],
        radius: int = 1,
        yaws: Optional[np.ndarray] = None,
) -> None:
    """
    Draw a trajectory on oriented arrow onto an RGB image
    Args:
        on_image (np.ndarray): the RGB image to draw onto
        positions (np.ndarray): pixel coordinates in the image space (not displacements) (Nx2)
        rgb_color (Tuple[int, int, int]): the trajectory RGB color
        radius (int): radius of the circle
        yaws (Optional[np.ndarray]): yaws in radians (N) or None to disable yaw visualisation

    Returns: None

    """
    if yaws is not None:
        assert len(yaws) == len(positions)
        for pos, yaw in zip(positions, yaws):
            pred_waypoint = pos[:2]
            pred_yaw = float(yaw[0])
            draw_arrowed_line(on_image, pred_waypoint, pred_yaw, rgb_color)
    else:
        for pos in positions:
            pred_waypoint = pos[:2]
            cv2.circle(on_image, tuple(pred_waypoint.astype(np.int)), radius, rgb_color, -1)


def draw_reference_trajectory(on_image: np.ndarray, world_to_pixel: np.ndarray, positions: np.ndarray) -> None:
    """
    Draw a trajectory (as points) onto the image
    Args:
        on_image (np.ndarray): the RGB image to draw onto
        world_to_pixel (np.ndarray): 3x3 matrix from meters to ego pixel space
        positions (np.ndarray): positions as 2D absolute meters coordinates

    Returns: None

    """
    positions_in_pixel_space = transform_points(np.array(positions), world_to_pixel)
    #  we clip the positions to be within the image
    mask = np.all(positions_in_pixel_space > (0.0, 0.0), 1) * np.all(positions_in_pixel_space < on_image.shape[:2], 1)
    positions_in_pixel_space = positions_in_pixel_space[mask]
    for pos in positions_in_pixel_space:
        cv2.circle(on_image, tuple(np.floor(pos).astype(np.int32)), 1, REFERENCE_TRAJ_COLOR, -1)


def draw_path_prior_layer(shape: Tuple, positions: np.ndarray, width: float = 3, vary_positions_len: bool = False) -> np.ndarray:
    """
    Return a path prior layer with a trajectory (as polyline)
    Args:
        on_image (np.ndarray): the RGB image to draw onto
        positions (np.ndarray): pixel coordinates in the image space (not displacements) (Nx2)
        width (float): thickness of the polyline edges
    Returns: np.ndarray

    """
    assert positions.ndim == 2

    if vary_positions_len:
        # Vary the length of reference trajectories
        col_len = positions.shape[0]
        random_len = randint(1,col_len)
        positions = positions[0:random_len, 0:]

    return cv2.polylines(np.zeros(shape, np.float32), [positions.astype(np.int32)], False, 1.0, width)
