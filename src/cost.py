import torch
from torch import nn


class CoFilLoss(nn.Module):
    def __init__(self, weight_decay: float = 0.0):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, F: torch.Tensor, Y: torch.Tensor, state_dict: dict):
        reg_value = 0.0
        for val in state_dict.values():
            if val.squeeze().ndim > 1:
                reg_value += val.pow(2).sum()

        return F.sub(Y).pow(2).sum().mul(0.5) + \
            (self.weight_decay / 2) * reg_value