import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any


class MoviesDataset(Dataset):
    def __init__(self, movies_file, ratings_file) -> None:
        super().__init__()

        self.movies_ = pd.read_csv(
            movies_file, header=0,  delimiter=',', quotechar='"')
        self.ratings_ = pd.read_csv(
            ratings_file, header=0,  delimiter=',', quotechar='"')

        self.user_count_ = self.ratings_['userId'].max()

    def __getitem__(self, index) -> Any:
        y = torch.zeros((1, self.user_count_), dtype=torch.float32)
        
        rated_users = self.ratings_.loc[
            self.ratings_['movieId'] == index + 1, ['userId', 'rating']
        ]

        y[0, (rated_users['userId'] - 1).tolist()] = torch.tensor(
            rated_users['rating'].tolist(), dtype=torch.float32)
        
        return y

    def __len__(self):
        return len(self.movies_)
