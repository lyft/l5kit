from typing import List, Optional

import numpy as np

from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import SemanticRasterizer


class SemBoxRasterizer(Rasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class
    """

    def __init__(
            self,
            render_context: RenderContext,
            filter_agents_threshold: float,
            history_num_frames: int,
            semantic_map_path: str,
            world_to_ecef: np.ndarray,
    ):
        super(SemBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        self.box_rast = BoxRasterizer(render_context, filter_agents_threshold, history_num_frames)
        self.sem_rast = SemanticRasterizer(render_context, semantic_map_path, world_to_ecef)

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        im_out_sem = self.sem_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        return np.concatenate([im_out_box, im_out_sem], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        total_ch = np.shape(in_im)[-1]
        pp_num_ch = 1 if "path_prior" in kwargs and kwargs["path_prior"] is True else 0
        sem_num_ch = 3
        box_num_ch = total_ch - sem_num_ch - pp_num_ch
        im_out_box = self.box_rast.to_rgb(in_im[..., :box_num_ch], **kwargs)
        im_out_sem = self.sem_rast.to_rgb(in_im[..., box_num_ch:(box_num_ch + sem_num_ch)], **kwargs)
        # merge the two together
        mask_box = np.any(im_out_box > 0, -1)
        im_out_sem[mask_box] = im_out_box[mask_box]

        if pp_num_ch:
            path_prior_mask = in_im[:,:,-1]>0.0
            im_out_sem[:,:,0][path_prior_mask] = 255
            im_out_sem[:,:,1][path_prior_mask] = 0
            im_out_sem[:,:,2][path_prior_mask] = 0

        return im_out_sem

    def num_channels(self) -> int:
        return self.box_rast.num_channels() + self.sem_rast.num_channels()
