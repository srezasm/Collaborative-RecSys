import math
import torch
from torch import nn


class CoFilModel(nn.Module):
    def __init__(self, num_users: int, num_movies: int, num_features: int) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.empty(
            num_movies, num_features, dtype=torch.float32))
        self.X = nn.Parameter(torch.empty(
            num_users, num_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.empty(
            1, num_users, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.X)
        nn.init.kaiming_uniform_(self.W)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self):
        return torch.mm(self.X, self.W.T).add(self.b)
