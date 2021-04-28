import numpy as np
import pytest

from l5kit.data import ChunkedDataset, filter_agents_by_frames, LocalDataManager, TL_FACE_DTYPE
from l5kit.rasterization import build_rasterizer, Rasterizer
from l5kit.sampling import get_history_slice


def check_rasterizer(cfg: dict, rasterizer: Rasterizer, zarr_dataset: ChunkedDataset) -> None:
    frames = zarr_dataset.frames[:]  # Load all frames into memory
    for current_frame in [0, 50, len(frames) - 1]:
        history_num_frames = cfg["model_params"]["history_num_frames"]
        s = get_history_slice(current_frame, history_num_frames, 1, include_current_state=True)
        frames_to_rasterize = frames[s]
        agents = filter_agents_by_frames(frames_to_rasterize, zarr_dataset.agents)
        tl_faces = [np.empty(0, dtype=TL_FACE_DTYPE) for _ in agents]  # TODO TR_FACES
        im = rasterizer.rasterize(frames_to_rasterize, agents, tl_faces)
        assert len(im.shape) == 3
        assert im.shape[-1] == rasterizer.num_channels()
        assert im.shape[:2] == tuple(cfg["raster_params"]["raster_size"])
        assert im.max() <= 255
        assert im.min() >= 0
        assert im.dtype == np.uint8

        rgb_im = rasterizer.to_rgb(im)
        assert im.shape[:2] == rgb_im.shape[:2]
        assert rgb_im.shape[2] == 3  # RGB has three channels
        assert rgb_im.dtype == np.uint8


@pytest.mark.parametrize("map_type", ["py_semantic", "py_satellite", "box_debug"])
def test_rasterizer_created_from_config(
        map_type: str, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    cfg["raster_params"]["map_type"] = map_type
    rasterizer = build_rasterizer(cfg, dmg)
    check_rasterizer(cfg, rasterizer, zarr_dataset)
