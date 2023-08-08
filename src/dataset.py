import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any


# TODO: movie/user filtering

class MovieLensDataset(Dataset):
    def __init__(self, movies_file, ratings_file) -> None:
        super().__init__()

        self.movies_df = pd.read_csv(
            movies_file, header=0,  delimiter=',', quotechar='"')
        self.ratings_df = pd.read_csv(
            ratings_file, header=0,  delimiter=',', quotechar='"')

        # remove unrated movies
        rated_movie_ids = self.ratings_df['movieId'].unique()
        rated_movie_ids.sort()
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(rated_movie_ids)]

        # replace movieId with a continues list
        ids_map = dict(
            zip(rated_movie_ids, range(len(rated_movie_ids)))
        )
        self.movies_df['movieId'] = self.movies_df['movieId'].replace(ids_map)
        self.ratings_df['movieId'] = self.ratings_df['movieId'].replace(ids_map)

        self.num_users = self.ratings_df['userId'].max()

        print('num_users:', self.num_users)
        print('num movies', len(self.movies_df))
        print('min movieId', self.movies_df['movieId'].min())
        print('max movieId', self.movies_df['movieId'].max())

    def __getitem__(self, index) -> Any:
        y = torch.zeros(self.num_users, dtype=torch.float32)

        rated_users = self.ratings_df.loc[
            self.ratings_df['movieId'] == index, ['userId', 'rating']
        ]

        y[(rated_users['userId'] - 1).tolist()] = torch.tensor(
            rated_users['rating'].tolist(), dtype=torch.float32)

        mean = y.mean(axis=2)

        y_norm = y - mean

        return y_norm

    def __len__(self):
        return len(self.movies_df)