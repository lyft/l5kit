from typing import List, Optional

import numpy as np

from .labels import PERCEPTION_LABEL_TO_INDEX, TL_FACE_LABEL_TO_INDEX


# Labels that belong to "agents" of some sort.
PERCEPTION_LABELS_TO_KEEP = [
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
]
PERCEPTION_LABEL_INDICES_TO_KEEP = [PERCEPTION_LABEL_TO_INDEX[label] for label in PERCEPTION_LABELS_TO_KEEP]

def _get_label_filter(label_probabilities: np.ndarray, threshold: float) -> np.array:
    """

    Arguments:
        label_probabilities (np.ndarray): Given the probabilities of all labels, returns a binary mask
        of those whose summed probability of the classes we are interested in is higher than the given threshold.

        This set of classes "we are interested in" is hardcoded for now.

    Keyword Arguments:
        threshold (float): probability threshold for filtering

    Returns:
        np.array -- A binary array which can be used to mask agents.
    """
    return np.sum(label_probabilities[:, PERCEPTION_LABEL_INDICES_TO_KEEP], axis=1) > threshold


def filter_agents_by_labels(agents: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Filters an agents array, keeping those agents that meet the threshold.

    Arguments:
        agents (np.ndarray): Agents array

    Keyword Arguments:
        threshold (float): probability threshold for filtering (default: {0.5})

    Returns:
        np.ndarray -- A subset of input ``agents`` array.
    """
    label_indices = _get_label_filter(agents["label_probabilities"], threshold)
    return agents[label_indices]


def filter_agents_by_track_id(agents: np.ndarray, track_id: int) -> np.ndarray:
    """Return all agent object (np.ndarray) of a given track_id.

    Arguments:
        agents (np.ndarray): agents array.
            NOTE: do NOT pass a zarr to this function, it can't handle boolean indexing
        track_id (int): agent track id to select

    Returns:
        np.ndarray -- Selected agent.
    """
    return agents[agents["track_id"] == track_id]


def filter_agents_by_frames(frames: np.ndarray, agents: np.ndarray) -> List[np.ndarray]:
    """
    Get a list of agents array, one array per frame. Note that "agent_index_interval" is used to filter agents,
    so you should take care of re-setting it if you have previously sliced agents.

    Args:
        frames (np.ndarray): an array of frames
        agents (np.ndarray): an array of agents

    Returns:
        List[np.ndarray] with the agents divided by frame
    """
    if frames.shape == ():
        frames = frames[None]  # add and axis if a single frame is passed
    return [agents[get_agents_slice_from_frames(frame)] for frame in frames]


def filter_tl_faces_by_frames(frames: np.ndarray, tl_faces: np.ndarray) -> List[np.ndarray]:
    """
    Get a list of traffic light faces arrays, one array per frame.
    This functions mimics `filter_agents_by_frames` for traffic light faces

    Args:
        frames (np.ndarray): an array of frames
        tl_faces (np.ndarray): an array of traffic light faces

    Returns:
        List[np.ndarray] with the traffic light faces divided by frame
    """
    return [tl_faces[get_tl_faces_slice_from_frames(frame)] for frame in frames]


def filter_tl_faces_by_status(tl_faces: np.ndarray, status: str) -> np.ndarray:
    """
    Filter tl_faces and keep only active faces
    Args:
        tl_faces (np.ndarray): array of traffic faces
        status (str): status we want to keep TODO refactor for enum

    Returns:
        np.ndarray: traffic light faces array with only faces with that status
    """
    return tl_faces[tl_faces["traffic_light_face_status"][:, TL_FACE_LABEL_TO_INDEX[status]] > 0]


def get_frames_slice_from_scenes(scene_a: np.ndarray, scene_b: Optional[np.ndarray] = None) -> slice:
    """
    Get a slice for indexing frames giving a start and end scene

    Args:
        scene_a (np.ndarray): the starting scene
        scene_b (Optional[np.ndarray]): the ending scene. If None, then scene_a end will be used

    Returns:
        slice: a slice object starting from the first frame in scene_a to the last one in scene_b
    """
    frame_index_start = scene_a["frame_index_interval"][0]
    if scene_b is None:
        scene_b = scene_a
    frame_index_end = scene_b["frame_index_interval"][1]
    return slice(frame_index_start, frame_index_end)


def get_agents_slice_from_frames(frame_a: np.ndarray, frame_b: Optional[np.ndarray] = None) -> slice:
    """
    Get a slice for indexing agents giving a start and end frame

    Args:
        frame_a (np.ndarray): the starting frame
        frame_b (Optional[np.ndarray]): the ending frame. If None, then frame_a end will be used

    Returns:
        slice: a slice object starting from the first agent in frame_a to the last one in frame_b
    """
    agent_index_start = frame_a["agent_index_interval"][0]
    if frame_b is None:
        frame_b = frame_a
    agent_index_end = frame_b["agent_index_interval"][1]
    return slice(agent_index_start, agent_index_end)


def get_tl_faces_slice_from_frames(frame_a: np.ndarray, frame_b: Optional[np.ndarray] = None) -> slice:
    """
    Get a slice for indexing traffic light faces giving a start and end frame

    Args:
        frame_a (np.ndarray): the starting frame
        frame_b (Optional[np.ndarray]): the ending frame. If None, then frame_a end will be used

    Returns:
        slice: a slice object starting from the first tl_face in frame_a to the last one in frame_b
    """
    tl_faces_index_start = frame_a["traffic_light_faces_index_interval"][0]
    if frame_b is None:
        frame_b = frame_a
    tl_faces_index_end = frame_b["traffic_light_faces_index_interval"][1]
    return slice(tl_faces_index_start, tl_faces_index_end)
