import bisect
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from zarr import convenience

from ..data import ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes
from ..kinematic import Perturbation
from ..rasterization import Rasterizer
from .ego import EgoDataset
from .select_agents import select_agents, TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE


# WARNING: changing these values impact the number of instances selected for both train and inference!
MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 1  # minimum number of frames an agents must have in the future to be picked


class AgentDataset(EgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
            render_path_prior: bool = False,
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(AgentDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation, render_path_prior)
        if agents_mask is None:  # if not provided try to load it from the zarr
            agents_mask = self.load_agents_mask()
            past_mask = agents_mask[:, 0] >= min_frame_history
            future_mask = agents_mask[:, 1] >= min_frame_future
            agents_mask = past_mask * future_mask

            if min_frame_history != MIN_FRAME_HISTORY:
                warnings.warn(
                    f"you're running with custom min_frame_history of {min_frame_history}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if min_frame_future != MIN_FRAME_FUTURE:
                warnings.warn(
                    f"you're running with custom min_frame_future of {min_frame_future}", RuntimeWarning, stacklevel=2
                )
        else:
            warnings.warn("you're running with a custom agents_mask", RuntimeWarning, stacklevel=2)

        # store the valid agents indices (N_valid_agents,)
        self.agents_indices = np.nonzero(agents_mask)[0]

        # store an array where valid indices have increasing numbers and the rest is -1 (N_total_agents,)
        self.mask_indices = agents_mask.copy().astype(np.int)
        self.mask_indices[self.mask_indices == 0] = -1
        self.mask_indices[self.mask_indices == 1] = np.arange(0, np.sum(agents_mask))

        # this will be used to get the frame idx from the agent idx
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        self.agents_mask = agents_mask

    def load_agents_mask(self) -> np.ndarray:
        """
        Loads a boolean mask of the agent availability stored into the zarr. Performs some sanity check against cfg.
        Returns: a boolean mask of the same length of the dataset agents

        """
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]

        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            warnings.warn(
                f"cannot find the right config in {self.dataset.path},\n"
                f"your cfg has loaded filter_agents_threshold={agent_prob};\n"
                "but that value doesn't have a match among the agents_mask in the zarr\n"
                "Mask will now be generated for that parameter.",
                RuntimeWarning,
                stacklevel=2,
            )

            select_agents(
                self.dataset,
                agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )

        agents_mask = convenience.load(str(agents_mask_path))  # note (lberg): this doesn't update root
        return agents_mask

    def __len__(self) -> int:
        """
        length of the available and reliable agents (filtered using the mask)
        Returns: the length of the dataset

        """
        return len(self.agents_indices)

    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent by iterating on agents and not AV.
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        index = self.agents_indices[index]
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, track_id=track_id)

    def get_scene_dataset(self, scene_index: int) -> "AgentDataset":
        """
        Differs from parent only in the return type.
        Instead of doing everything from scratch, we rely on super call and fix the agents_mask
        """

        new_dataset = super(AgentDataset, self).get_scene_dataset(scene_index).dataset

        # filter agents_bool values
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        # ASSUMPTION: all agents_index are consecutive
        start_index = self.dataset.frames[frame_interval[0]]["agent_index_interval"][0]
        end_index = self.dataset.frames[frame_interval[1] - 1]["agent_index_interval"][1]
        agents_mask = self.agents_mask[start_index:end_index].copy()

        return AgentDataset(
            self.cfg, new_dataset, self.rasterizer, self.perturbation, agents_mask,  # overwrite the loaded one
            self.render_path_prior
        )

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        frame_slice = get_frames_slice_from_scenes(scenes[scene_idx])
        agent_slice = get_agents_slice_from_frames(*self.dataset.frames[frame_slice][[0, -1]])

        mask_valid_indices = (self.agents_indices >= agent_slice.start) * (self.agents_indices < agent_slice.stop)
        indices = np.nonzero(mask_valid_indices)[0]
        return indices

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        assert frame_idx < len(self.dataset.frames), f"frame_idx {frame_idx} is over len {len(self.dataset.frames)}"

        # avoid using `get_agents_slice_from_frames` as it hits the disk
        agent_start = self.cumulative_sizes_agents[frame_idx - 1] if frame_idx > 0 else 0
        agent_end = self.cumulative_sizes_agents[frame_idx]
        # slice using frame boundaries and take only valid indices
        mask_idx = self.mask_indices[agent_start:agent_end]
        indices = mask_idx[mask_idx != -1]
        return indices
