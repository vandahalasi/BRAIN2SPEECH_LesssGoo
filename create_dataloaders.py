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


def get_data(data_path):
    """
    Loads features and spectograms into SpectogramDataset and creates train, 
    validation and test dataloaders.
    """

    participants = ['sub-%02d'%i for i in range(1,11)]

    feat_names = np.load(os.path.join(feat_path,f'{participants[0]}_feat_names.npy'))
    feat_name_dict = {feat : idx for idx, feat in enumerate(feat_names)}
    for ptcp in participants:
        feat_names = np.load(os.path.join(feat_path,f'{ptcp}_feat_names.npy'))
        for feat_name in feat_names:
            if feat_name not in feat_name_dict.keys():
                feat_name_dict[feat_name] = len(feat_name_dict)

    for idx, ptcp in enumerate(participants[:4]):
        spec_ptcp = np.load(os.path.join(feat_path,f'{ptcp}_spec.npy'))
        feat_ptcp = np.load(os.path.join(feat_path,f'{ptcp}_feat.npy'))
        feat_names = np.load(os.path.join(feat_path,f'{ptcp}_feat_names.npy'))
        right_dim_feat = np.zeros((len(feat_ptcp),len(feat_name_dict)))
        for i, feat_name in enumerate(feat_names):
            j = feat_name_dict[feat_name]
            right_dim_feat[:,j] = feat_ptcp[:,i]

        train_spec, val_spec, test_spec = np.split(spec_ptcp, [int(len(spec_ptcp)*0.8), 
                int(len(spec_ptcp)*0.85)])
        train_feat, val_feat, test_feat = np.split(right_dim_feat, [int(len(right_dim_feat)*0.8), 
                int(len(right_dim_feat)*0.85)])

        if idx == 0:
            spectogram_train = train_spec
            spectogram_val = val_spec
            spectogram_test = test_spec
            features_train = train_feat
            features_val = val_feat
            features_test = test_feat

        spectogram_train = np.concatenate((spectogram_train, train_spec), axis=0)
        spectogram_val = np.concatenate((spectogram_val, val_spec), axis=0)
        spectogram_test = np.concatenate((spectogram_test, test_spec), axis=0)
        features_train = np.concatenate((features_train, train_feat), axis=0)
        features_val = np.concatenate((features_val, val_feat), axis=0)
        features_test = np.concatenate((features_test, test_feat), axis=0)

    return spectogram_train, spectogram_val, spectogram_test, features_train, features_val, features_test

def create_datasets(spectogram_train, spectogram_val, spectogram_test, features_train, features_val, features_test, window=3):
    #create a Dataset
    train_dataset = SpectogramDataset(features_train, spectogram_train, window)
    val_dataset = SpectogramDataset(features_val, spectogram_val, window)
    test_dataset = SpectogramDataset(features_test, spectogram_test, window)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    #create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, eval_loader, test_loader

def main():
    spectogram_train, spectogram_val, spectogram_test, features_train, features_val, features_test = get_data()
    # Only need to be run if train_stats.json is missing
    # write_statistics_to_json(features_train)
    train_dataset, val_dataset, test_dataset = create_datasets(spectogram_train, spectogram_val, spectogram_test, features_train, features_val, features_test, window=3)
    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32)
 
