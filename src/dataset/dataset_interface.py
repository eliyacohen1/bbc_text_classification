import torch
import numpy as np


class DatasetInterface(torch.utils.data.Dataset):

    def __len__(self):
        return len(self.labels)

    def get_labels(self, idx):
        return np.array(self.labels[idx])

    def get_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        texts = self.get_texts(idx)
        y = self.get_labels(idx)

        return texts, y
