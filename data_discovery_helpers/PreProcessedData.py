from data_discovery_helpers.models import PreProcessedDataModel
import numpy as np
import os

from typing import Tuple
import matplotlib.pyplot as plt

DEFAULT_FEATURES_PATH = r'./assets/features'

class PreProcessedData:
    __preprocessed_data_model:PreProcessedDataModel
    
    def __init__(self,participant_id: str, features_path:str):
            spec = np.load(os.path.join(features_path,f'{participant_id}_spec.npy'))
            feat = np.load(os.path.join(features_path,f'{participant_id}_feat.npy'))

            self.__preprocessed_data_model = PreProcessedDataModel(features=feat,spectogram=spec,p_id=participant_id)

    def split_data(self,train_ratio: float,eval_ratio: float)->Tuple[PreProcessedDataModel,PreProcessedDataModel,PreProcessedDataModel]:
        if(train_ratio+eval_ratio>1):
            raise ValueError("Invalid train_ratio eval_ratio combination.")

        spec_ptcp = self.__preprocessed_data_model.spectogram
        feat_ptcp = self.__preprocessed_data_model.features
        p_id = self.__preprocessed_data_model.p_id

        train_spec, val_spec, test_spec = np.split(spec_ptcp, [int(len(spec_ptcp)*train_ratio), 
                                                    int(len(spec_ptcp)*(train_ratio+eval_ratio))])
        train_feat, val_feat, test_feat = np.split(feat_ptcp, [int(len(feat_ptcp)*train_ratio), 
                                                    int(len(feat_ptcp)*(train_ratio+eval_ratio))])
                                        
        train_ds =  PreProcessedDataModel(p_id = p_id,
                                        features= train_feat,
                                        spectogram= train_spec)                  
        
        val_ds =  PreProcessedDataModel(p_id = p_id,
                                        features= val_feat,
                                        spectogram= val_spec) 
        
        test_ds =  PreProcessedDataModel(p_id = p_id,
                                        features= test_feat,
                                        spectogram= test_spec)
        
        return (train_ds,val_ds,test_ds)

    def display_spectogram(self):
        start_s = 5.5
        stop_s=19.5

        frameshift = 0.01
        spectrogram = self.__preprocessed_data_model.spectogram
        
        cm='viridis'
        fig, ax = plt.subplots(1, sharex=True)
        pSta=int(start_s*(1/frameshift));pSto=int(stop_s*(1/frameshift))
        ax.imshow(np.flipud(spectrogram[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
        ax.set_ylabel(f'Log Mel-Spec Bin of {self.__preprocessed_data_model.p_id}')
        
        

    def display_feature_map(self):
        pass
    
    def get_model(self)->PreProcessedDataModel:
        return self.__preprocessed_data_model