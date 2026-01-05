import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy
import math

class Dataset:
    """
    Abstract Class for Datasets.
    """
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Yield batches
        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                continue
                
            batch_indices = indices[start_idx:end_idx]
            
            # Fetch Batch
            # We assume dataset supports list indexing or we loop
            batch = []
            # Optimization: Try to batch fetch if dataset supports it
            # But standard MapDataset is item by item
            
            # For list/array based datasets (common in this framework so far), we can optimize
            if isinstance(self.dataset, (list, np.ndarray)):
                 yield self.dataset[batch_indices] # Might fail if list
            elif hasattr(self.dataset, 'tensors'):
                 # TensorDataset optimization
                 # Use array indexing on the internal arrays
                 # We avoid calling __getitem__ N times
                 yield tuple(t[batch_indices] for t in self.dataset.tensors)
            else:
                 # Standard loop
                 batch = [self.dataset[i] for i in batch_indices]
                 
                 # Collate
                 # If batch is list of tuples, unzip
                 if isinstance(batch[0], tuple):
                     transposed = zip(*batch)
                     yield [np.stack(samples, axis=0) for samples in transposed]
                 else:
                     yield np.stack(batch, axis=0)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
