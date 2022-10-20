import torch
import json

class SpectogramDataset(torch.utils.data.Dataset):   
    def __init__(self, data, spectogram, window):
        self.target_spectogram = spectogram
        self.features = data
        self.window = window
        
        # Opening JSON file for mean and std of training set
        f = open("train_stats.json")
        stats = json.load(f)
        train_mean = stats["mean"]
        train_std = stats["std"]
        # Normalization
        self.features=(self.features-train_mean)/train_std


    def __len__(self):
        return self.features.shape[0]-self.window


    def __getitem__(self, index):
        return self.features[index:index+self.window,:], self.target_spectogram[index:index+self.window,:]