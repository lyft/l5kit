from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..data.filter import filter_tl_faces_by_status
from ..data.map_api import InterpolationMethod, MapAPI, TLFacesColors
from ..geometry import rotation33_as_yaw, transform_point, transform_points
from .rasterizer import Rasterizer
from .render_context import RenderContext


# sub-pixel drawing precision constants
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]
INTERPOLATION_POINTS = 20


class RasterEls(IntEnum):  # map elements
    LANE_NOTL = 0
    ROAD = 1
    CROSSWALK = 2


COLORS = {
    TLFacesColors.GREEN.name: (0, 255, 0),
    TLFacesColors.RED.name: (255, 0, 0),
    TLFacesColors.YELLOW.name: (255, 255, 0),
    RasterEls.LANE_NOTL.name: (255, 217, 82),
    RasterEls.ROAD.name: (17, 17, 31),
    RasterEls.CROSSWALK.name: (255, 117, 69),
}


def indices_in_bounds(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords


class SemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
            self, render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray,
    ):
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        self.mapAPI = MapAPI(semantic_map_path, world_to_ecef)

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_im

    def render_semantic_map(
            self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # get all lanes as interpolation so that we can transform them all together

        lane_indices = indices_in_bounds(center_in_world, self.mapAPI.bounds_info["lanes"]["bounds"], raster_radius)
        lanes_mask: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool))
        lanes_area = np.zeros((len(lane_indices) * 2, INTERPOLATION_POINTS, 2))

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]

            # interpolate over polyline to always have the same number of points
            lane_coords = self.mapAPI.get_lane_as_interpolation(
                lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
            lanes_area[idx * 2] = lane_coords["xyz_left"][:, :2]
            lanes_area[idx * 2 + 1] = lane_coords["xyz_right"][::-1, :2]

            lane_type = RasterEls.LANE_NOTL.name
            lane_tl_ids = set(self.mapAPI.get_lane_traffic_control_ids(lane_idx))
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                lane_type = self.mapAPI.get_color_for_face(tl_id)

            lanes_mask[lane_type][idx * 2: idx * 2 + 2] = True

        if len(lanes_area):
            lanes_area = cv2_subpixel(transform_points(lanes_area.reshape((-1, 2)), raster_from_world))

            for lane_area in lanes_area.reshape((-1, INTERPOLATION_POINTS * 2, 2)):
                # need to for-loop otherwise some of them are empty
                cv2.fillPoly(img, [lane_area], COLORS[RasterEls.ROAD.name], **CV2_SUB_VALUES)

            lanes_area = lanes_area.reshape((-1, INTERPOLATION_POINTS, 2))
            for name, mask in lanes_mask.items():  # draw each type of lane with its own color
                cv2.polylines(img, lanes_area[mask], False, COLORS[name], **CV2_SUB_VALUES)

        # plot crosswalks
        crosswalks = []
        for idx in indices_in_bounds(center_in_world, self.mapAPI.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.mapAPI.get_crosswalk_coords(self.mapAPI.bounds_info["crosswalks"]["ids"][idx])
            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, COLORS[RasterEls.CROSSWALK.name], **CV2_SUB_VALUES)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return in_im

    def num_channels(self) -> int:
        return 3
