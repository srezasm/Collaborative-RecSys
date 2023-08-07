import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MOVIES_CSV_FILE = '../ml-latest/movies.csv'
RATINGS_CSV_FILE = '../ml-latest/ratings.csv'