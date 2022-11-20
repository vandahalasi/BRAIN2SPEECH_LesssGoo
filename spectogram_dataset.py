import torch
import json
import numpy as np

class SpectogramDataset(torch.utils.data.Dataset):   
    def __init__(self, data, spectogram, window):
        self.target_spectogram = np.array(spectogram, dtype=np.float32)
        self.features = data
        self.window = window
        
        # Opening JSON file for mean and std of training set
        f = open("train_stats.json")
        stats = json.load(f)
        self.train_mean = stats["mean"]
        self.train_std = np.array(stats["std"])
        

    def __len__(self):
        return self.features.shape[0]-self.window


    def __getitem__(self, index):
        # Normalization
        input = (self.features[index:index+self.window,:]-self.train_mean)/(self.train_std + 1e-7)
        return input.astype(np.float32), self.target_spectogram[index:index+self.window,:]