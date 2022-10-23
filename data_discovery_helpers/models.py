from pydantic import BaseModel
from enum import Enum
import numpy as np
from typing import Any, Optional

class SexEnum(str,Enum):
    FEMALE = "Female"
    MALE = "Male"

class ParticipantModel(BaseModel):
    p_idx: int
    participant_id: str
    age: int
    sex: SexEnum
    hand: Any
    

class NWBDataModel(BaseModel):
    p_id: str
    eeg: np.ndarray
    eeg_sr: int
    audio: np.ndarray
    audio_sr: int
    channels: np.ndarray
    words: np.ndarray
    
    class Config:
        arbitrary_types_allowed = True

class PreProcessedDataModel(BaseModel):
    p_id: str
    features: np.ndarray 
    spectogram: np.ndarray

    def __add__(self, other):
        if not isinstance(other,PreProcessedDataModel):
            raise AttributeError("InvalidAttribute")
        
        concated_feat = np.concatenate((self.features,other.features), axis=0)
        concated_spectogram = np.concatenate((self.spectogram,other.spectogram),axis=0)

        return PreProcessedDataModel(p_id="Combinated",features=concated_feat,spectogram=concated_spectogram)

    class Config:
        arbitrary_types_allowed = True 