import warnings
from typing import Tuple

import numpy as np

from ..geometry import rotation33_as_yaw, yaw_as_rotation33
from ..random import RandomGenerator
from .ackerman_steering_model import fit_ackerman_model_exact
from .perturbation import Perturbation


def _get_trajectory(past_frames: np.ndarray, future_frames: np.ndarray) -> np.ndarray:
    """A function that takes past & future frames, extracts ego translation & rotation from the frames, sorts them
    based on time (from past to future) then return sorted ego translation & rotation in a trajectory array.
    :param past_frames: a list of past frames
    :type past_frames: np.ndarray
    :param future_frames: a list of future frames
    :type future_frames: np.ndarray
    :return: a ego trajectory (position + heading) sorted from past to future
    :rtype: np.ndarray
    """
    num_frames = len(past_frames) + len(future_frames)
    trajectory = np.zeros((num_frames, 3), dtype=np.float32)

    # Note that history frames go backward in time from the anchor frame.
    trajectory[:, :2] = np.concatenate(
        (past_frames["ego_translation"][::-1, :2], future_frames["ego_translation"][:, :2])
    )
    rotations = np.concatenate((past_frames["ego_rotation"][::-1], future_frames["ego_rotation"]))
    trajectory[:, 2] = [rotation33_as_yaw(rot) for rot in rotations]

    return trajectory


class AckermanPerturbation(Perturbation):
    def __init__(
        self,
        random_offset_generator: RandomGenerator,
        perturb_prob: float,
        min_displacement: float = 0,
    ):
        """
        Apply Ackerman to get a feasible trajectory with probability perturb_prob.

        :param random_offset_generator: a callable that yields 3 values (lat, long, yaw offsets)
        :param perturb_prob: probability between 0 and 1 of applying the perturbation
        :param min_displacement: minimum displacement required to apply lateral & yaw perturbation
        """
        self.perturb_prob = perturb_prob
        self.min_displacement = min_displacement
        self.random_offset_generator = random_offset_generator
        if perturb_prob == 0:
            warnings.warn(
                "Consider replacing this object with None if no perturbation is intended", RuntimeWarning, stacklevel=2
            )

    def perturb(self, history_frames: np.ndarray, future_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        num_history_frames = len(history_frames)
        num_future_frames = len(future_frames)

        assert num_history_frames > 0, f"Number of history frames: {num_history_frames}"
        assert num_future_frames > 0, f"Number of future frames: {num_future_frames}"

        trajectory = _get_trajectory(history_frames, future_frames)

        curr_frame_idx = num_history_frames - 1
        displacements = np.linalg.norm(np.diff(trajectory[curr_frame_idx:, :2], axis=0), axis=1)

        if num_history_frames > 1:
            curr_displacement = np.linalg.norm(np.diff(trajectory[curr_frame_idx - 1:curr_frame_idx + 1, :2], axis=0))
        else:
            curr_displacement = displacements[0]

        if np.random.rand() >= self.perturb_prob:
            return history_frames.copy(), future_frames.copy(), curr_displacement

        (
            lateral_offset_m,
            longitudinal_offset_m,
            yaw_offset_rad,
            speed_multiplier,
            speed_offset,
        ) = self.random_offset_generator()
        speed_multiplier = max(0.2, min(speed_multiplier, 1.8))

        # TODO: ackerman lateral & yaw perturbation does not work when EGO slow moving
        if np.sum(displacements) < self.min_displacement:
            lateral_offset_m = 0
            yaw_offset_rad = 0

        position_offset_m = np.array([longitudinal_offset_m, lateral_offset_m, 0])
        position_offset_m = np.matmul(yaw_as_rotation33(trajectory[curr_frame_idx, 2]), position_offset_m)

        #  perform ackerman steering model fitting
        gx = trajectory[curr_frame_idx + 1:, 0]
        gy = trajectory[curr_frame_idx + 1:, 1]
        gr = trajectory[curr_frame_idx + 1:, 2]
        gv = displacements

        x0 = trajectory[curr_frame_idx, 0] + position_offset_m[0]
        y0 = trajectory[curr_frame_idx, 1] + position_offset_m[1]
        r0 = trajectory[curr_frame_idx, 2] + yaw_offset_rad
        v0 = curr_displacement * speed_multiplier + speed_offset

        wgx = np.ones(num_future_frames)
        wgy = np.ones(num_future_frames)
        wgr = np.ones(num_future_frames)
        wgv = np.zeros(num_future_frames)

        new_xs, new_ys, new_yaws, new_vs, new_acc, new_steer = fit_ackerman_model_exact(
            x0, y0, r0, v0, gx, gy, gr, gv, wgx, wgy, wgr, wgv,
        )

        history_frames["ego_translation"][0, 0] = x0
        history_frames["ego_translation"][0, 1] = y0
        history_frames["ego_rotation"][0] = yaw_as_rotation33(r0)

        future_frames["ego_translation"][:, 0] = new_xs
        future_frames["ego_translation"][:, 1] = new_ys
        future_frames["ego_rotation"] = np.array([yaw_as_rotation33(yaw) for yaw in new_yaws])

        return history_frames, future_frames, v0
