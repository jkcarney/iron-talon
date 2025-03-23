import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Dota2Dataset(Dataset):
    def __init__(self, h5_fp, indices=None, positional_encoding_dim=16):
        """
        Args:
            h5_fp (str): Path to the HDF5 file.
            indices (list[int] or ndarray): Indices for this dataset split.
            positional_encoding_dim (int): Dimension of positional encoding.
        """
        super().__init__()
        
        # Open the file
        self.file = h5py.File(h5_fp, 'r')
        self.matches = self.file['tensors']  # shape: (N, 24, 125)
        self.labels = self.file['labels']    # shape: (N,)
        self.num_draft_steps = self.matches.shape[1]  # 24
        self.positional_encoding_dim = positional_encoding_dim

        # If no indices provided, use all
        if indices is None:
            self.indices = np.arange(self.matches.shape[0])
        else:
            self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the "idx in [0..len(indices))" to the actual data index
        data_index = self.indices[idx]

        # 1. Randomly pick a step from 0..23
        step_idx = np.random.randint(0, self.num_draft_steps)

        # 2. Extract hero picks for that step => shape: (125,)
        hero_slice = self.matches[data_index, step_idx]
        hero_slice = torch.tensor(hero_slice, dtype=torch.float)

        # 3. Label => 0 or 1
        label = self.labels[data_index]
        label = torch.tensor(label, dtype=torch.long)

        # 4. Positional encoding
        pos_encoding = self._sinusoidal_encoding(step_idx, self.positional_encoding_dim)

        return hero_slice, pos_encoding, label

    def _sinusoidal_encoding(self, position, d_model):
        """
        Standard sinusoidal positional encoding of dimension d_model.
        """
        encoding = np.zeros(d_model, dtype=np.float32)
        for i in range(0, d_model, 2):
            encoding[i] = np.sin(position / (10000 ** (i / d_model)))
            if (i + 1) < d_model:
                encoding[i + 1] = np.cos(position / (10000 ** ((i + 1) / d_model)))
        return torch.tensor(encoding, dtype=torch.float)
