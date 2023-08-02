import torch
import numpy as np
import os

CACHE_DIR = 'cache'


def load_check(file_path: str):
    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f'Directory {CACHE_DIR} doesn\'t exits')
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File {file_path} doesn\'t exits')


def save_check():
    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)


def save_np_array(arr: np.ndarray, file_name: str) -> str:
    file_path = os.path.join('.', CACHE_DIR, file_name)

    save_check()

    np.savez_compressed(file_path, arr)

    return f'Saved array in {file_path}'


def load_np_arr(file_name: str) -> np.ndarray:
    file_path = os.path.join('.', CACHE_DIR, file_name)

    load_check(file_path)

    return np.load(file_path)['arr_0']


def save_tensor(tensor: torch.Tensor, file_name: str) -> str:
    file_path = os.path.join('.', CACHE_DIR, file_name)

    save_check()

    torch.save(tensor, file_path)

    return f'Saved tensor in {file_path}'


def load_tensor(file_name: str):
    file_path = os.path.join('.', CACHE_DIR, file_name)

    load_check(file_path)

    return torch.load(file_path)
