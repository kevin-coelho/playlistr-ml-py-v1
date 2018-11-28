from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SpotifyDataset(Dataset):

    def __init__(self, data_arr, labels, transform=None):
        # self.set = pd.DataFrame(data=data_arr)
        # self.labels = pd.DataFrame(data=labels)
        # self.transform = transform
        self.set = data_arr
        self.labels = labels
        self.transform = transform
    #
    # def __len__(self):
    #     return len(self.set)
    def __len__(self):
        m,n = self.set.shape
        return m

    # def __getitem__(self, idx):
    #     sample = {'data': self.set.iloc[idx, 1:].as_matrix(), 'labels': self.labels.iloc[idx].as_matrix()}
    #
    #     if self.transform:
    #         sample = self.transform(sample)
    def __getitem__(self, idx):
        sample = {'data': self.set[idx, :], 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
