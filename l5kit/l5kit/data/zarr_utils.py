import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from . import ChunkedDataset

GIGABYTE = 1 * 1024 * 1024 * 1024


def _compute_path_size(path: str) -> int:
    """
    Compute the total size of the folder, considering also nested elements.
    Can be run to get zarr total size

    Args:
        path (str): base path

    Returns:
        (int): total size in bytes
    """
    root_directory = Path(path)
    return sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())


def _get_num_els_in_scene_range(zarr_dataset: ChunkedDataset, scene_index_start: int, scene_index_end: int) -> dict:
    """
    Get numbers of scenes, frames, agents, tl_lights in a set of scene in a zarr
    Args:
        zarr_dataset (ChunkedDataset): zarr dataset to use for computing number of elements
        scene_index_start (int): start from this scene (included)
        scene_index_end (int): end before this scene (excluded!!)

    Returns:
        dict: a dict with keys equal to zarr initialise args
    """
    assert scene_index_end > scene_index_start

    scene_start = zarr_dataset.scenes[scene_index_start]
    scene_end = zarr_dataset.scenes[scene_index_end - 1]

    frame_start = zarr_dataset.frames[scene_start["frame_index_interval"][0]]
    frame_end = zarr_dataset.frames[scene_end["frame_index_interval"][1] - 1]

    return {
        "num_scenes": scene_index_end - scene_index_start,
        "num_frames": scene_end["frame_index_interval"][1] - scene_start["frame_index_interval"][0],
        "num_agents": frame_end["agent_index_interval"][1] - frame_start["agent_index_interval"][0],
        "num_tl_faces": frame_end["traffic_light_faces_index_interval"][1]
        - frame_start["traffic_light_faces_index_interval"][0],
    }


def _append_zarr_subset(
    input_zarr: ChunkedDataset,
    output_zarr: ChunkedDataset,
    scene_index_start: int,
    scene_index_end: int,
    output_zarr_num_els: Optional[dict] = None,
) -> None:
    """
    Append a subset of input_zarr into output_zarr. To avoid appending (slow), output_zarr must be opened in write mode
    and with pre-allocated shape already. End indices of output_zarr are read from output_zarr_num_els, or 0 is assumed
    otherwise

    Args:
        input_zarr (ChunkedDataset): origin zarr in read mode
        output_zarr (ChunkedDataset): zarr already opened in write mode and with pre-allocated arrays
        scene_index_start (int): index of the first scene to copy
        scene_index_end (int): index of the last scene (excluded)
        output_zarr_num_els (Optional[dict]): if None, write starting from 0 index in the output zarr

    Returns:

    """

    # indices to assign in the destination array
    if output_zarr_num_els is None:
        idx_output_scene, idx_output_frame, idx_output_agent, idx_output_tl_face = 0, 0, 0, 0
    else:
        idx_output_scene = output_zarr_num_els["num_scenes"]
        idx_output_frame = output_zarr_num_els["num_frames"]
        idx_output_agent = output_zarr_num_els["num_agents"]
        idx_output_tl_face = output_zarr_num_els["num_tl_faces"]

    # relative indices to subtract before copying to erase input history
    idx_start_frame = input_zarr.scenes[scene_index_start]["frame_index_interval"][0]
    idx_start_agent = input_zarr.frames[idx_start_frame]["agent_index_interval"][0]
    idx_start_tl_face = input_zarr.frames[idx_start_frame]["traffic_light_faces_index_interval"][0]

    # if output_zarr_num_els is not zero we also need to add output_history
    idx_start_frame = idx_output_frame - idx_start_frame
    idx_start_agent = idx_output_agent - idx_start_agent
    idx_start_tl_face = idx_output_tl_face - idx_start_tl_face

    for idx_scene in range(scene_index_start, scene_index_end):
        # get slices from input zarr
        scenes = input_zarr.scenes[idx_scene : idx_scene + 1]
        frames = input_zarr.frames[slice(*scenes[0]["frame_index_interval"])]
        agents = input_zarr.agents[slice(frames[0]["agent_index_interval"][0], frames[-1]["agent_index_interval"][1])]
        tl_faces = input_zarr.tl_faces[
            slice(
                frames[0]["traffic_light_faces_index_interval"][0], frames[-1]["traffic_light_faces_index_interval"][1]
            )
        ]

        # fix indices
        scenes["frame_index_interval"] += idx_start_frame
        frames["agent_index_interval"] += idx_start_agent
        frames["traffic_light_faces_index_interval"] += idx_start_tl_face

        # copy from input_zarr to output_zarr
        output_zarr.scenes[idx_output_scene : idx_output_scene + len(scenes)] = scenes
        output_zarr.frames[idx_output_frame : idx_output_frame + len(frames)] = frames
        output_zarr.agents[idx_output_agent : idx_output_agent + len(agents)] = agents
        output_zarr.tl_faces[idx_output_tl_face : idx_output_tl_face + len(tl_faces)] = tl_faces

        # update output indices
        idx_output_scene += len(scenes)
        idx_output_frame += len(frames)
        idx_output_agent += len(agents)
        idx_output_tl_face += len(tl_faces)


def zarr_concat(input_zarrs: List[str], output_zarr: str) -> None:
    """
    Concat many zarr into a single one. Takes care of updating indices for frames and agents.

    Args:
        input_zarrs (List[str]): a list of paths to input zarrs
        output_zarr (str): the path to the output zarr

    Returns:

    """

    assert not os.path.exists(output_zarr), "we need to pre-allocate zarr, can't append fast"
    output_dataset = ChunkedDataset(output_zarr)

    # we need to estimate how much to allocate by reading all input zarrs lens
    # we also store them for later use
    num_els_valid_zarrs = []
    valid_zarrs = []

    tqdm_bar = tqdm(input_zarrs, desc="computing total size to allocate")
    for input_zarr in tqdm_bar:
        try:
            input_dataset = ChunkedDataset(input_zarr)
            input_dataset.open()
        except (ValueError, KeyError):
            print(f"{input_zarr} is not valid! skipping")
            continue
        num_els_valid_zarrs.append(_get_num_els_in_scene_range(input_dataset, 0, len(input_dataset.scenes)))
        valid_zarrs.append(input_zarr)

    # we can now pre-allocate the output dataset
    total_num_els: Counter = Counter()
    for num_el in num_els_valid_zarrs:
        total_num_els += Counter(num_el)
    output_dataset.initialize(**total_num_els)

    cur_num_els = Counter({"num_scenes": 0, "num_frames": 0, "num_agents": 0, "num_tl_faces": 0})
    tqdm_bar = tqdm(valid_zarrs)
    for idx, input_zarr in enumerate(tqdm_bar):
        tqdm_bar.set_description(f"working on {input_zarr}")

        input_dataset = ChunkedDataset(input_zarr)
        input_dataset.open()

        _append_zarr_subset(input_dataset, output_dataset, 0, len(input_dataset.scenes), cur_num_els)
        cur_num_els += Counter(num_els_valid_zarrs[idx])


def zarr_split(input_zarr: str, output_path: str, split_infos: List[dict]) -> List[Tuple[int, int]]:
    """
    Split the input zarr into many zarrs. Names and sizes can be passed using the split_infos arg.

    Args:
        input_zarr (str): path of the original zarr
        output_path (str): base destination path
        split_infos (List[dict]): list of dict. Each element should have `name` (final path is output_path+name)
        and `split_size_GB` with the size of the split. Last element must have `split_size_GB` set to -1 to collect
        the last part of the input_zarrr.

    Returns:
        List[Tuple[int, int]]: indices of scenes where a split occurred
    """
    input_dataset = ChunkedDataset(input_zarr)
    input_dataset.open()

    assert len(split_infos) > 0
    assert split_infos[-1]["split_size_GB"] == -1, "last split element should have split_size_GB equal to -1"

    # compute the size of the input_dataset in GB
    size_input_gb = _compute_path_size(input_zarr) / GIGABYTE
    num_scenes_input = len(input_dataset.scenes)

    # ensure the defined splits don't overspill the input dataset
    num_scenes_output = [
        int(num_scenes_input * split_info["split_size_GB"] / size_input_gb) for split_info in split_infos[:-1]
    ]
    assert sum(num_scenes_output) < num_scenes_input, "size exceed"
    num_scenes_output.append(num_scenes_input - sum(num_scenes_output))

    cur_scene = 0
    cuts_track = []  # keep track of start-end of the cuts
    tqdm_bar = tqdm(zip(num_scenes_output, split_infos))
    for num_scenes, split_info in tqdm_bar:
        start_cut = cur_scene
        end_cut = cur_scene + num_scenes
        tqdm_bar.set_description(f"cutting scenes {start_cut}-{end_cut} into {split_info['name']}")

        num_els_output = _get_num_els_in_scene_range(input_dataset, start_cut, end_cut)

        output_dataset = ChunkedDataset(str(Path(output_path) / split_info["name"]))
        output_dataset.initialize(**num_els_output)

        _append_zarr_subset(input_dataset, output_dataset, start_cut, end_cut)
        cuts_track.append((start_cut, end_cut))
        cur_scene = end_cut

    return cuts_track
