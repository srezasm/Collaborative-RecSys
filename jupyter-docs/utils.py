import numpy as np
import os

CACHE_DIR = 'cache'


def save_np_array(arr: np.ndarray, file_name: str) -> None:
    file_path = os.path.join('.' ,CACHE_DIR, file_name)

    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    np.savez_compressed(file_path, arr)

    return f'Saved array in {file_path}'



def load_np_arr(file_name: str) -> np.ndarray:
    file_path = os.path.join('.', CACHE_DIR, file_name)

    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError()
    if not os.path.isfile(file_path):
        raise FileNotFoundError()

    return np.load(file_path)['arr_0']
