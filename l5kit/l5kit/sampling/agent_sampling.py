from typing import List, Optional, Tuple

import numpy as np

from l5kit.visualization import draw_path_prior_layer

from ..data import (filter_agents_by_labels, filter_tl_faces_by_frames, get_agents_slice_from_frames,
                    get_tl_faces_slice_from_frames)
from ..data.filter import filter_agents_by_frames, filter_agents_by_track_id
from ..geometry import angular_distance, compute_agent_pose, rotation33_as_yaw, transform_points
from ..kinematic import Perturbation
from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from .slicing import get_future_slice, get_history_slice


def get_agent_context(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        history_num_frames: int,
        future_num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Slice zarr or numpy arrays to get the context around the agent onf interest (both in space and time)

    Args:
        state_index (int): frame index inside the scene
        frames (np.ndarray): frames from the scene
        agents (np.ndarray): agents from the scene
        tl_faces (np.ndarray): tl_faces from the scene
        history_num_frames (int): how many frames in the past to slice
        future_num_frames (int): how many frames in the future to slice

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
    """

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, 1, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, 1)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()
    # sync interval with the agents array
    history_frames["agent_index_interval"] -= agent_slice.start
    future_frames["agent_index_interval"] -= agent_slice.start
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    # get traffic lights (past and future)
    tl_slice = get_tl_faces_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    tl_faces = tl_faces[tl_slice].copy()
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    future_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces)
    future_tl_faces = filter_tl_faces_by_frames(future_frames, tl_faces)

    return history_frames, future_frames, history_agents, future_agents, history_tl_faces, future_tl_faces


def compute_agent_velocity(
        history_positions_m: np.ndarray, future_positions_m: np.ndarray, step_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute estimated velocities by finite differentiation on future positions as(pos(T+t) - pos(T))/t.
    This simple approach gives less than 0.5% velocity difference
    compared to (pos(T+t) - pos(T-t))/2t on v1.1/sample.zarr.tar.

    Args:
        history_positions_m (np.ndarray): history XY positions in meters
        future_positions_m (np.ndarray): future XY positions in meters
        step_time (np.ndarray): length of a step in second

    Returns:
        Tuple[np.ndarray, np.ndarray]: history and future XY speeds

    """
    assert step_time > np.finfo(float).eps, f"step_time must be greater then eps, got {step_time}"

    # [future_num_frames, 2]
    future_positions_diff_m = np.concatenate((future_positions_m[:1], np.diff(future_positions_m, axis=0)))
    # [future_num_frames, 2]
    future_vels_mps = np.float32(future_positions_diff_m / step_time)

    # current position is included in history positions
    # [history_num_frames, 2]
    history_positions_diff_m = np.diff(history_positions_m, axis=0)
    # [history_num_frames, 2]
    history_vels_mps = np.float32(history_positions_diff_m / step_time)

    return history_vels_mps, future_vels_mps


def compute_signed_speeds(vels_mps: np.ndarray) -> np.ndarray:
    """ Compute signed speeds by computing the 2-norm of the given velocities, the sign / direction of the speed is
    based on the sign of longitudinal velocity.

    :param vels_mps: input velocity with shape [num_timestamps, 2]
    :return: computed signed speed with shape [num_timestamps]
    """
    signed_speeds_mps = np.sign(vels_mps[:, 0]) * np.linalg.norm(vels_mps, axis=1)
    return signed_speeds_mps


def get_relative_poses(
        num_frames: int,
        frames: np.ndarray,
        selected_track_id: Optional[int],
        agents: List[np.ndarray],
        agent_from_world: np.ndarray,
        current_agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).

    Note: return dtype is float32, even if the provided args are float64. Still, the transformation
    between space is performed in float64 to ensure precision

    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, extent, availabilities

    """
    # How much the coordinates differ from the current state in meters.
    positions_m = np.zeros((num_frames, 2), dtype=agent_from_world.dtype)
    yaws_rad = np.zeros((num_frames, 1), dtype=np.float32)
    extents_m = np.zeros((num_frames, 2), dtype=np.float32)
    availabilities = np.zeros((num_frames,), dtype=np.float32)

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
            agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)
        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
                agent_centroid_m = agent["centroid"]
                agent_yaw_rad = agent["yaw"]
                agent_extent = agent["extent"][:2]
            except IndexError:
                availabilities[i] = 0.0  # keep track of invalid futures/history
                continue

        positions_m[i] = agent_centroid_m
        yaws_rad[i] = agent_yaw_rad
        extents_m[i] = agent_extent
        availabilities[i] = 1.0

    # batch transform to speed up
    positions_m = transform_points(positions_m, agent_from_world) * availabilities[:, np.newaxis]
    yaws_rad = angular_distance(yaws_rad, current_agent_yaw) * availabilities[:, np.newaxis]
    return positions_m.astype(np.float32), yaws_rad, extents_m, availabilities


def generate_agent_sample(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        selected_track_id: Optional[int],
        render_context: RenderContext,
        history_num_frames: int,
        future_num_frames: int,
        step_time: float,
        filter_agents_threshold: float,
        rasterizer: Optional[Rasterizer] = None,
        perturbation: Optional[Perturbation] = None,
        render_path_prior: bool = False,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        render_context (RenderContext): The context for rasterisation
        history_num_frames (int): Amount of history frames to draw into the rasters
        future_num_frames (int): Amount of history frames to draw into the rasters
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data
        render_path_prior bool: if to draw a path prior in a new layer of image

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames, future_num_frames, )

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        filtered_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
        agent = filter_agents_by_track_id(filtered_agents, selected_track_id)[0]
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)
    raster_from_agent = raster_from_world @ world_from_agent

    future_positions_m, future_yaws_rad, future_extents, future_availabilities = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad,
    )
    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_extents, history_availabilities = get_relative_poses(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
    )

    history_vels_mps, future_vels_mps = compute_agent_velocity(history_positions_m, future_positions_m, step_time)
    future_speeds_mps = compute_signed_speeds(future_vels_mps)

    if input_im is not None and render_path_prior is True:
        future_positions_avail_m = future_positions_m[future_availabilities == 1]
        path_prior_layer = draw_path_prior_layer(input_im.shape[:2], raster_from_agent, future_positions_avail_m,
                                                 vary_positions_len=True)
        input_im = np.concatenate([input_im, path_prior_layer[..., None]], axis=2)

    result = {
        "frame_index": state_index,
        "image": input_im,
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_speeds": future_speeds_mps,
        "target_speed": future_speeds_mps[:1],   # TODO deprecate
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m,
        "history_yaws": history_yaws_rad,
        "history_velocities": history_vels_mps,
        "history_availabilities": history_availabilities,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "history_extents": history_extents,
        "future_extents": future_extents,
    }
    if len(history_vels_mps) > 0:
        # estimated current speed based on displacement between current frame at T and past frame at T-1
        result["curr_speed"] = np.linalg.norm(history_vels_mps[0])
    return result
