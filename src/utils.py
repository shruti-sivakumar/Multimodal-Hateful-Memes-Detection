import random
import numpy as np
import torch
from dataclasses import dataclass


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    embedding_dim: int = 512
    num_classes: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 10
    max_length: int = 128
    subset_size: int = 100
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True