from typing import List, Optional

import cv2
import numpy as np

from ..geometry import rotation33_as_yaw, transform_point
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .satellite_image import get_sat_image_crop_scaled


class SatelliteRasterizer(Rasterizer):
    """This rasterizer takes a satellite image in its constructor and a transform from world coordinates to this image.
    When you call rasterize, it will return a crop around the agent of interest with the agent's forward vector
    pointing right for the current timestep.
    """

    def __init__(
            self,
            render_context: RenderContext,
            map_im: np.ndarray,
            world_to_aerial: np.ndarray,
            interpolation: int = cv2.INTER_LINEAR,
    ):
        """

        Arguments:
            render_context (RenderContext): Render Context
            map_im (np.ndarray): Satellite image to crop from.
            world_to_aerial (np.ndarray): Transform to go from map coordinates to satellite image pixel coordinates.
        """
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.map_im = map_im
        self.world_to_aerial = world_to_aerial
        self.interpolation = interpolation
        self.map_pixel_scale = (
            1 / np.linalg.norm(world_to_aerial[0, 0:3]) + 1 / np.linalg.norm(world_to_aerial[1, 0:3])
        ) / 2

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # Note 1: it looks like we are assuming that yaw in ecef == yaw in sat image
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])

        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # Transform raster center to satellite coordinates (consider also z here)
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)
        center_in_aerial_px = transform_point(np.append(center_in_world_m, ego_translation_m[2]), self.world_to_aerial)

        # Note 2: there is a negation here, unknown why this is necessary.
        # My best guess is because Y is flipped, maybe we can do this more elegantly.
        sat_im = get_sat_image_crop_scaled(
            self.map_im,
            self.raster_size,
            center_in_aerial_px,
            yaw=-ego_yaw_rad,
            pixel_size=self.pixel_size,
            sat_pixel_scale=self.map_pixel_scale,
            interpolation=self.interpolation,
        )
        if not self.render_context.set_origin_to_bottom:
            sat_im = sat_im[::-1]
        return sat_im

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return in_im

    def num_channels(self) -> int:
        return 3
