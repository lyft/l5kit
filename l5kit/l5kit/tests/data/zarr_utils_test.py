from pathlib import Path
from uuid import uuid4

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.zarr_utils import zarr_concat, zarr_split


def test_zarr_concat(dmg: LocalDataManager, tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    concat_count = 4
    zarr_input_path = dmg.require("single_scene.zarr")
    zarr_output_path = str(tmp_path / f"{uuid4()}.zarr")

    zarr_concat([zarr_input_path] * concat_count, zarr_output_path)
    zarr_cat_dataset = ChunkedDataset(zarr_output_path)
    zarr_cat_dataset.open()

    # check lens of arrays
    assert len(zarr_cat_dataset.scenes) == len(zarr_dataset.scenes) * concat_count
    assert len(zarr_cat_dataset.frames) == len(zarr_dataset.frames) * concat_count
    assert len(zarr_cat_dataset.agents) == len(zarr_dataset.agents) * concat_count
    assert len(zarr_cat_dataset.tl_faces) == len(zarr_dataset.tl_faces) * concat_count

    # check the first and last element concat_count times
    input_scene_a = zarr_dataset.scenes[0]
    input_scene_b = zarr_dataset.scenes[-1]
    input_frame_a = zarr_dataset.frames[0]
    input_frame_b = zarr_dataset.frames[-1]
    input_agent_a = zarr_dataset.agents[0]
    input_agent_b = zarr_dataset.agents[-1]
    input_tl_a = zarr_dataset.tl_faces[0]
    input_tl_b = zarr_dataset.tl_faces[-1]

    for idx in range(concat_count):
        output_scene_a = zarr_cat_dataset.scenes[idx * len(zarr_dataset.scenes)]
        output_scene_b = zarr_cat_dataset.scenes[(idx + 1) * len(zarr_dataset.scenes) - 1]

        # check all scene fields
        assert output_scene_a["host"] == input_scene_a["host"]
        assert output_scene_a["start_time"] == input_scene_a["start_time"]
        assert output_scene_a["end_time"] == input_scene_a["end_time"]

        displace_frame = len(zarr_dataset.frames) * idx
        assert np.all(output_scene_a["frame_index_interval"] == input_scene_a["frame_index_interval"] + displace_frame)
        assert np.all(output_scene_b["frame_index_interval"] == input_scene_b["frame_index_interval"] + displace_frame)

        # check all the frame fields
        output_frame_a = zarr_cat_dataset.frames[idx * len(zarr_dataset.frames)]
        output_frame_b = zarr_cat_dataset.frames[(idx + 1) * len(zarr_dataset.frames) - 1]

        assert np.allclose(output_frame_a["ego_rotation"], input_frame_a["ego_rotation"])
        assert np.allclose(output_frame_b["ego_rotation"], input_frame_b["ego_rotation"])
        assert np.allclose(output_frame_a["ego_translation"], input_frame_a["ego_translation"])
        assert np.allclose(output_frame_b["ego_translation"], input_frame_b["ego_translation"])
        assert output_frame_a["timestamp"] == input_frame_a["timestamp"]
        assert output_frame_b["timestamp"] == input_frame_b["timestamp"]

        displace_agent = len(zarr_dataset.agents) * idx
        assert np.all(output_frame_a["agent_index_interval"] == input_frame_a["agent_index_interval"] + displace_agent)
        assert np.all(output_frame_b["agent_index_interval"] == input_frame_b["agent_index_interval"] + displace_agent)

        displace_tl = len(zarr_dataset.tl_faces) * idx
        assert np.all(
            output_frame_a["traffic_light_faces_index_interval"]
            == input_frame_a["traffic_light_faces_index_interval"] + displace_tl
        )
        assert np.all(
            output_frame_b["traffic_light_faces_index_interval"]
            == input_frame_b["traffic_light_faces_index_interval"] + displace_tl
        )

        # check agents
        output_agent_a = zarr_cat_dataset.agents[idx * len(zarr_dataset.agents)]
        output_agent_b = zarr_cat_dataset.agents[(idx + 1) * len(zarr_dataset.agents) - 1]
        assert output_agent_a == input_agent_a
        assert output_agent_b == input_agent_b

        # check tfl
        output_tl_a = zarr_cat_dataset.tl_faces[idx * len(zarr_dataset.tl_faces)]
        output_tl_b = zarr_cat_dataset.tl_faces[(idx + 1) * len(zarr_dataset.tl_faces) - 1]
        assert output_tl_a == input_tl_a
        assert output_tl_b == input_tl_b


def test_zarr_split(dmg: LocalDataManager, tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    concat_count = 10
    zarr_input_path = dmg.require("single_scene.zarr")
    zarr_cat_path = str(tmp_path / f"{uuid4()}.zarr")
    zarr_concat([zarr_input_path] * concat_count, zarr_cat_path)

    split_infos = [
        {"name": f"{uuid4()}.zarr", "split_size_GB": 0.002},  # cut around 2MB
        {"name": f"{uuid4()}.zarr", "split_size_GB": 0.001},  # cut around 0.5MB
        {"name": f"{uuid4()}.zarr", "split_size_GB": -1},
    ]  # everything else

    scene_splits = zarr_split(zarr_cat_path, str(tmp_path), split_infos)

    # load the zarrs and check elements
    zarr_cat = ChunkedDataset(zarr_cat_path)
    zarr_cat.open()

    for scene_split, split_info in zip(scene_splits, split_infos):
        zarr_out = ChunkedDataset(str(tmp_path / str(split_info["name"])))
        zarr_out.open()

        # compare elements at the start and end of each scene in both zarrs
        for idx_scene in range(len(zarr_out.scenes)):

            # compare elements in the scene
            input_scene = zarr_cat.scenes[scene_split[0] + idx_scene]
            input_frames = zarr_cat.frames[slice(*input_scene["frame_index_interval"])]
            input_agents = zarr_cat.agents[
                input_frames[0]["agent_index_interval"][0] : input_frames[-1]["agent_index_interval"][1]
            ]
            input_tl_faces = zarr_cat.tl_faces[
                input_frames[0]["traffic_light_faces_index_interval"][0] : input_frames[-1][
                    "traffic_light_faces_index_interval"
                ][1]
            ]

            output_scene = zarr_out.scenes[idx_scene]
            output_frames = zarr_out.frames[slice(*output_scene["frame_index_interval"])]
            output_agents = zarr_out.agents[
                output_frames[0]["agent_index_interval"][0] : output_frames[-1]["agent_index_interval"][1]
            ]
            output_tl_faces = zarr_out.tl_faces[
                output_frames[0]["traffic_light_faces_index_interval"][0] : output_frames[-1][
                    "traffic_light_faces_index_interval"
                ][1]
            ]

            assert np.all(input_frames["ego_translation"] == output_frames["ego_translation"])
            assert np.all(input_frames["ego_rotation"] == output_frames["ego_rotation"])
            assert np.all(input_agents == output_agents)
            assert np.all(input_tl_faces == output_tl_faces)
