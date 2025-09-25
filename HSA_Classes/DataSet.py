# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import torch as t
import numpy as np
from torch.utils.data import Dataset

from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_dataset(Dataset):

    def __init__(self, preprocessed_np):
        assert isinstance(
            preprocessed_np, np.ndarray
        ), f"preprocessed_np must be an np.ndarray, not {type(preprocessed_np)}"
        self.preprocessed_t = t.tensor(preprocessed_np)  # pass dtype here.
        self.logger = logger

    def __len__(self):
        self.logger.trace("Dataset __len__")
        return len(self.preprocessed_t)

    def __getitem__(self, idx):
        data = self.preprocessed_t[idx]
        self.logger.trace("Dataset __getitem__")
        return data
