import pandas as pd

from models import ParticipantModel,SexEnum


class Participant:
    __participant_model: ParticipantModel

    def __init__(self,pd_row: pd.core.series.Series,p_idx: int):
        sex = SexEnum.MALE
        if(pd_row[2]=='F'):
            sex = SexEnum.FEMALE
        
        self.__participant_model = ParticipantModel(p_idx=p_idx,
                                            participant_id=pd_row[0],
                                            age=pd_row[1],
                                            sex=sex,
                                            hand=pd_row[3] 
                                        )
    def get_model(self)->ParticipantModel:
        return self.__participant_model                        
