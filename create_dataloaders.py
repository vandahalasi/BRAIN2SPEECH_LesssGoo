import os

import numpy as np
import json
from torch.utils.data import DataLoader

from spectogram_dataset import SpectogramDataset


def write_statistics_to_json(data):
    """
    Saving mean and std of training data into a json file for normalization.
    """
    stats = {}
    mean = np.mean(data,axis=0)
    std=np.std(data,axis=0)
    stats["mean"] = mean.tolist()
    stats["std"] = std.tolist()
    # Serializing json
    json_object = json.dumps(stats, indent=4)
    
    # Writing to sample.json
    with open("train_stats.json", "w") as outfile:
        outfile.write(json_object)


def get_dataloaders(batch_size=32):
    """
    Loads features and spectograms into SpectogramDataset and creates train, 
    validation and test dataloaders.
    """
    feat_path = r'./SingleWordProductionDutch/features'

    participants = ['sub-%02d'%i for i in range(1,11)]

    #Load the data
    spec_ptcp = np.load(os.path.join(feat_path,f'{participants[0]}_spec.npy'))
    feat_ptcp = np.load(os.path.join(feat_path,f'{participants[0]}_feat.npy'))
    spectogram_train, spectogram_val, spectogram_test = np.split(spec_ptcp, 
                [int(len(spec_ptcp)*0.8), int(len(spec_ptcp)*0.85)])
    features_train, features_val, features_test = np.split(feat_ptcp, 
                [int(len(feat_ptcp)*0.8), int(len(feat_ptcp)*0.85)])

    for ptcp in participants[1:2]:
        spec_ptcp = np.load(os.path.join(feat_path,f'{ptcp}_spec.npy'))
        feat_ptcp = np.load(os.path.join(feat_path,f'{ptcp}_feat.npy'))

        train_spec, val_spec, test_spec = np.split(spec_ptcp, [int(len(spec_ptcp)*0.8), 
                int(len(spec_ptcp)*0.85)])
        train_feat, val_feat, test_feat = np.split(feat_ptcp, [int(len(feat_ptcp)*0.8), 
                int(len(feat_ptcp)*0.85)])

        spectogram_train = np.concatenate((spectogram_train, train_spec), axis=0)
        spectogram_val = np.concatenate((spectogram_val, val_spec), axis=0)
        spectogram_test = np.concatenate((spectogram_test, test_spec), axis=0)
        features_train = np.concatenate((features_train, train_feat), axis=0)
        features_val = np.concatenate((features_val, val_feat), axis=0)
        features_test = np.concatenate((features_test, test_feat), axis=0)

    # write_statistics_to_json(features_train)

    #create a Dataset
    train_dataset = SpectogramDataset(features_train, spectogram_train)
    val_dataset = SpectogramDataset(features_val, spectogram_val)
    test_dataset = SpectogramDataset(features_test, spectogram_test)

    #create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, eval_loader, test_loader
