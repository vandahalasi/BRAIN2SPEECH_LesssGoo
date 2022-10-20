import torch
import json

class SpectogramDataset(torch.utils.data.Dataset):   
    def __init__(self, data, spectogram):
        self.target_spectogram = spectogram
        self.features = data
        
        # Opening JSON file for mean and std of training set
        f = open("train_stats.json")
        stats = json.load(f)
        train_mean = stats["mean"]
        train_std = stats["std"]
        # Normalization
        self.features=(self.features-train_mean)/train_std


    def __len__(self):
        return self.features.shape[0]-3


    def __getitem__(self, index):
        return self.features[index:index+3,:], self.target_spectogram[index:index+3,:]