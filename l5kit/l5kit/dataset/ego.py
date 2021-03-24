import bisect
import warnings
from functools import partial
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ..data import (ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes,
                    get_tl_faces_slice_from_frames)
from ..kinematic import Perturbation
from ..rasterization import Rasterizer, RenderContext
from ..sampling import generate_agent_sample
from .utils import convert_str_to_fixed_length_tensor


class EgoDataset(Dataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            render_path_prior: bool = False,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
None if not desired
        """
        self.perturbation = perturbation
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.rasterizer = rasterizer
        self.render_path_prior = render_path_prior

        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
        )

        # build a partial so we don't have to access cfg each time
        self.sample_function = partial(
            generate_agent_sample,
            render_context=render_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            step_time=cfg["model_params"]["step_time"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
            render_path_prior=render_path_prior
        )

    def __len__(self) -> int:
        """
        Get the number of available AV frames

        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames)

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
            (position and yaw) along with their availability, the 2D matrix to center that agent,
            the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        try:
            if self.cfg["raster_params"]["disable_traffic_light_faces"]:
                tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        except KeyError:
            warnings.warn(
                "disable_traffic_light_faces not found in config, this will raise an error in the future",
                RuntimeWarning,
                stacklevel=2,
            )
        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        # add information only, so that all data keys are always preserved
        data["scene_index"] = scene_index
        data["host_id"] = convert_str_to_fixed_length_tensor(self.dataset.scenes[scene_index]["host"])
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch
        data["world_to_image"] = data["raster_from_world"]  # TODO deprecate

        # when rast is None, image could be None. In that case we remove the key
        if data["image"] is not None:
            data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        else:
            del data["image"]

        return data

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

    def get_scene_dataset(self, scene_index: int) -> "EgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        # copy everything to avoid references (scene is already detached from zarr if get_combined_scene was called)
        scenes = self.dataset.scenes[scene_index: scene_index + 1].copy()
        frame_slice = get_frames_slice_from_scenes(*scenes)
        frames = self.dataset.frames[frame_slice].copy()
        agent_slice = get_agents_slice_from_frames(*frames[[0, -1]])
        tl_slice = get_tl_faces_slice_from_frames(*frames[[0, -1]])

        agents = self.dataset.agents[agent_slice].copy()
        tl_faces = self.dataset.tl_faces[tl_slice].copy()

        frames["agent_index_interval"] -= agent_slice.start
        frames["traffic_light_faces_index_interval"] -= tl_slice.start
        scenes["frame_index_interval"] -= frame_slice.start

        dataset = ChunkedDataset("")
        dataset.agents = agents
        dataset.tl_faces = tl_faces
        dataset.frames = frames
        dataset.scenes = scenes

        return EgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation, self.render_path_prior)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. EgoDataset iterates over frames, so this is just a matter
        of finding the scene boundaries.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        frames = self.dataset.frames
        assert frame_idx < len(frames), f"frame_idx {frame_idx} is over len {len(frames)}"
        return np.asarray((frame_idx,), dtype=np.int64)

    def __str__(self) -> str:
        return self.dataset.__str__()
