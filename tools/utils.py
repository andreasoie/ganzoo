import random

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(lucky_number: int = 1337):
    random.seed(lucky_number)
    np.random.seed(lucky_number)
    torch.manual_seed(lucky_number)
