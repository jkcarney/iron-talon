import numpy as np



def split_indices(n, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Return three arrays: train_idx, val_idx, test_idx
    such that they partition the range [0..n).
    """
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = int(train_ratio * n + val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx
