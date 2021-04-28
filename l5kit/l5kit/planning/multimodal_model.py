from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PlanningModel


class MultiModalPlanningModel(PlanningModel):
    def __init__(
        self,
        model_arch: str,
        num_timestamps: int,
        num_outputs: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        num_modes: int,
        pretrained: bool = True,
        coef_ce: float = 0.5,
    ) -> None:
        num_targets = (num_timestamps * num_outputs + 1) * num_modes
        super().__init__(
            model_arch, num_targets, weights_scaling, criterion, pretrained
        )
        self.num_modes = num_modes
        self.num_timestamps = num_timestamps
        self.num_outputs = num_outputs
        self.coef_ce = coef_ce

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        batch_size = len(image_batch)

        # [batch_size, (num_timestamps * num_outputs * num_modes) + num_modes]
        outputs_all = self.model(image_batch)

        # [batch_size, num_timestamps * num_outputs * num_modes]
        outputs = outputs_all[:, :-self.num_modes]
        # [batch_size, num_modes]
        outputs_nll = outputs_all[:, -self.num_modes:]

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_timestamps * num_modes, num_outputs]
            outputs = outputs.view(batch_size, self.num_timestamps * self.num_modes, self.num_outputs)

            # [batch_size, num_timestamps, 2]
            xy = data_batch["target_positions"]
            # [batch_size, num_timestamps, 1]
            yaw = data_batch["target_yaws"]

            # [batch_size, num_timestamps, num_outputs]
            targets = torch.cat((xy, yaw), dim=-1)
            # [batch_size, num_timestamps]
            target_weights = data_batch["target_availabilities"] > 0.5
            # [batch_size, num_timestamps, num_outputs]
            target_weights = target_weights.unsqueeze(-1) * self.weights_scaling
            # [batch_size, num_timestamps * num_modes, num_outputs]
            losses = self.criterion(
                outputs, targets.repeat(1, self.num_modes, 1)
            ) * target_weights.repeat(1, self.num_modes, 1)
            # [batch_size, num_modes]
            cost_dist = losses.view(batch_size, self.num_modes, -1).mean(dim=-1)
            # [batch_size,]
            assignment = cost_dist.argmin(dim=-1)
            # [1]
            loss_dist = cost_dist[torch.arange(batch_size, device=outputs.device), assignment].mean()
            # [1]
            loss_nll = F.cross_entropy(outputs_nll, assignment)
            return {"loss": loss_dist + self.coef_ce * loss_nll}
        else:
            outputs = outputs.view(batch_size, self.num_modes, self.num_timestamps, self.num_outputs)
            outputs_selected = outputs[torch.arange(batch_size, device=outputs.device), outputs_nll.argmax(-1)]
            pred_positions, pred_yaws = outputs_selected[..., :2], outputs_selected[..., 2:3]
            pred_pos_all = outputs[..., :2]
            pred_yaw_all = outputs[..., 2:3]
            return {
                "positions": pred_positions,
                "yaws": pred_yaws,
                "positions_all": pred_pos_all,
                "yaws_all": pred_yaw_all,
            }
