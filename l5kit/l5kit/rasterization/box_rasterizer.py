from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE

from ..data.filter import filter_agents_by_labels, filter_agents_by_track_id
from ..geometry import rotation33_as_yaw, transform_points
from .rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import CV2_SUB_VALUES, cv2_subpixel


def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:  # TODO this can be useful to have around
    """ Get a valid agent with information from the AV. Ford Fusion extent is used.

    :param frame: The frame from which the Ego states are extracted
    :return An agent numpy array of the Ego states
    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


def get_box_world_coords(agents: np.ndarray) -> np.ndarray:
    """Get world coordinates of the 4 corners of the bounding boxes

    :param agents: agents array of size N with centroid (world coord), yaw and extent
    :return: array of shape (N, 4, 2) with the four corners of each agent
    """
    # shape is (1, 4, 2)
    corners_base_coords = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5)[None, :, :]

    # compute the corner in world-space (start in origin, rotate and then translate)
    # extend extent to shape (N, 1, 2) so that op gives (N, 4, 2)
    corners_m = corners_base_coords * agents["extent"][:, None, :2]  # corners in zero
    s = np.sin(agents["yaw"])
    c = np.cos(agents["yaw"])
    # note this is clockwise because it's right-multiplied and not left-multiplied later,
    # and therefore we're still rotating counterclockwise.
    rotation_m = np.moveaxis(np.array(((c, s), (-s, c))), 2, 0)
    # extend centroid to shape (N, 1, 2) so that op gives (N, 4, 2)
    box_world_coords = corners_m @ rotation_m + agents["centroid"][:, None, :2]
    return box_world_coords


def draw_boxes(
        raster_size: Tuple[int, int],
        raster_from_world: np.ndarray,
        agents: np.ndarray,
        color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """ Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected on the image space.
    Finally, cv2 draws the boxes.

    :param raster_size: Desired output raster image size
    :param raster_from_world: Transformation matrix to transform from world to image coordinate
    :param agents: An array of agents to be drawn
    :param color: Single int or RGB color
    :return: the image with agents rendered. A RGB image if using RGB color, otherwise a GRAY image
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    box_world_coords = get_box_world_coords(agents)
    box_raster_coords = transform_points(box_world_coords.reshape((-1, 2)), raster_from_world)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_raster_coords = cv2_subpixel(box_raster_coords.reshape((-1, 4, 2)))
    cv2.fillPoly(im, box_raster_coords, color=color, **CV2_SUB_VALUES)
    return im


class BoxRasterizer(Rasterizer):
    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        render_ego_history: bool = True,
    ) -> None:
        """This is a rasterizer class used for rendering agents' bounding boxes on the raster image.

        :param render_context: Render context
        :param filter_agents_threshold: Value between 0 and 1 used to filter uncertain agent detections
        :param history_num_frames: Number of past frames to be renderd on the raster
        :param render_ego_history: Option to render past ego states on the raster image
        """
        super(BoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames
        self.render_ego_history = render_ego_history

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate raster image by rendering Ego & Agents as bounding boxes on the raster image.
        Ego & Agents from different past frame are rendered at different image channel.

        :param history_frames: A list of past frames to be rasterized
        :param history_agents: A list of agents from past frames to be rasterized
        :param history_tl_faces: A list of traffic light faces from past frames to be rasterized
        :param agent: The selected agent to be rendered as Ego, if it is None the AV will be rendered as Ego
        :return: An raster image of size [2xN] with Ego & Agents rendered as bounding boxes, where N is number of
         history frames
        """
        # all frames are drawn relative to this one"
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.raster_size[1], self.raster_size[0], self.history_num_frames + 1)
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                ego_agent = av_agent
            else:
                ego_agent = filter_agents_by_track_id(agents, agent["track_id"])
                agents = np.append(agents, av_agent)  # add av_agent to agents
                if len(ego_agent) > 0:  # check if ego_agent is in the frame
                    agents = agents[agents != ego_agent[0]]  # remove ego_agent from agents

            agents_images[..., i] = draw_boxes(self.raster_size, raster_from_world, agents, 255)
            if len(ego_agent) > 0 and (self.render_ego_history or i == 0):
                ego_images[..., i] = draw_boxes(self.raster_size, raster_from_world, ego_agent, 255)

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images), -1)

        return out_im

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """This function is used to get an rgb image where agents further in the past have faded colors.

        :param in_im: The output of the rasterize function
        :param kwargs: This can be used for additional customization (such as colors)
        :return: An RGB image with agents and ego coloured with fading colors
        """
        hist_frames = in_im.shape[-1] // 2
        in_im = np.transpose(in_im, (2, 0, 1)) / 255

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im = (np.clip(out_im_agent + out_im_ego, 0, 1) * 255).astype(np.uint8)
        return out_im

    def num_channels(self) -> int:
        return (self.history_num_frames + 1) * 2
