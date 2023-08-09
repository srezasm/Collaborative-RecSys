import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Any


class MovieLensDataset(Dataset):
    def __init__(self, movies_file, ratings_file, rated_movie_ids: list) -> None:
        super().__init__()

        self.movies_df = pd.read_csv(
            movies_file, header=0,  delimiter=',', quotechar='"')

        # Filter the movies with rated genres
        genres = set()
        for movie_id in rated_movie_ids:
            cur_genres = self.movies_df.loc[
                self.movies_df['movieId'] == movie_id, 'genres'
            ].str.split('|').explode()

            genres.update(cur_genres)

        self.movies_df = self.movies_df[
            self.movies_df['genres'].str.contains('|'.join(genres))
        ]

        # ignore other movie ratings
        # remove the users that haven't rated any of the selected movies
        self.ratings_df = pd.read_csv(
            ratings_file, header=0,  delimiter=',', quotechar='"')

        self.ratings_df = self.ratings_df[
            self.ratings_df['movieId'].isin(self.movies_df['movieId'])
        ]        

        # re-index users
        

        self.num_users = len(self.ratings_df['userId'].unique())
        self.num_movies = len(self.movies_df)

    def __getitem__(self, index) -> Any:
        y = torch.zeros(self.num_users)

        movie_id = self.movies_df.iloc[index, 0]

        rated_users = self.ratings_df.loc[
            self.ratings_df['movieId'] == movie_id, ['userId', 'rating']
        ]

        # print(len(rated_users['rating']))         76813
        # print(len(rated_users['userId'] - 1))     76813
        # print(y.shape)                            316725
        y[(rated_users['userId'] - 1).tolist()] = torch.tensor(rated_users['rating'].tolist())

        mean = y.mean()

        y_norm = y - mean

        return y_norm

    def __len__(self):
        return self.num_movies
