import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import os


import matplotlib.pyplot as plt
from data_discovery_helpers.models import NWBDataModel


class NWBData:
    __nwb_data_model: NWBDataModel

    def __init__(self,participant_id: str, path_bids: str):
        io = NWBHDF5IO(os.path.join(path_bids,participant_id,'ieeg',f'{participant_id}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        #channels
        channels = pd.read_csv(os.path.join(path_bids,participant_id,'ieeg',f'{participant_id}_task-wordProduction_channels.tsv'), delimiter='\t')
        channels = np.array(channels['name'])
        
        self.__nwb_data_model = NWBDataModel(p_id = participant_id,
                                            eeg = eeg,
                                            eeg_sr = eeg_sr,
                                            audio = audio,
                                            audio_sr = audio_sr,
                                            channels = channels,
                                            words = words)
    def display_eeg_signal(self,channel_to_display: int):
        """Displays one channel EEG time series signal.
        
            Args:
                channels_to_display(int): EEG channel to display.
                    Must be between zero and data.shape[1] (channel_num) 

            Raises:
                ValueError: if the channel_to_display argument is invalid. 
        """
        data_model = self.__nwb_data_model
        if (channel_to_display<1 or channel_to_display>data_model.eeg.shape[1]):
            raise ValueError("Invalid channel num.")
        
        Ts = 1/data_model.eeg_sr
        time = np.arange(0,data_model.eeg.shape[0],1)*Ts
        channel = data_model.eeg[:,channel_to_display]
        
        plt.plot(time,channel)
        plt.title(f"EEG Signal Participant id: {data_model.p_id}, Channel: {channel_to_display}") 
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [mV]")
    
    
    def create_word_histogram(self):
        pass
    
    def get_data(self)->NWBDataModel:
        return self.__nwb_data_model