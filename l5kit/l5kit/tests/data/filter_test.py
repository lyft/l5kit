import numpy as np
import pytest

from l5kit.data import ChunkedDataset, filter_agents_by_frames


@pytest.mark.parametrize("frame_bound", [0, 10, 50, 100])
def test_get_frames_agents_shape(frame_bound: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert len(agents) == frame_bound


@pytest.mark.parametrize("frame_bound", [pytest.param(0, marks=pytest.mark.xfail), 10, 50, 100])
def test_get_frames_agents_ret(frame_bound: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert sum([len(agents_fr) for agents_fr in agents]) > 0


@pytest.mark.parametrize("frame_idx", [0, 10, 50, 100])
def test_get_frames_agents_single_frame(frame_idx: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[frame_idx], zarr_dataset.agents)
    assert len(agents) == 1
    assert isinstance(agents[0], np.ndarray)
