import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(lucky_number: int = 3407):
    random.seed(lucky_number)
    np.random.seed(lucky_number)
    torch.manual_seed(lucky_number)


def init_outdir(file_name: str, name: str) -> str:
    outdir = os.path.join(os.path.dirname(os.path.abspath(file_name)), name)
    os.makedirs(outdir, exist_ok=True)
    return outdir
