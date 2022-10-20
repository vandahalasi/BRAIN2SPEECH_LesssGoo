# Team LesssGoo #

## Participants ##
* Halasi Vanda Réka (HL382A)
* Seres Zsombor
* Harsányi Dániel

## BRAIN2SPEECH Project ##
The aim of this project is to generate audio files based on brain signes using Machine Learning based algorithms. We are using the following [article](https://www.nature.com/articles/s41597-022-01542-9) as a reference. The preprocessing steps such as filtering and channel selection have been taken from their work, which can be found in [SingleWordProductionDutch](https://github.com/neuralinterfacinglab/SingleWordProductionDutch/tree/28fb2d2db4c3332ba95f831208ffb5dd3dcde223) submodule.


## Installation guide: ##
- [ ] Navigate to root directory
- [ ] pip install -r requirements.txt : installs dependencies
- [ ] python setup.py build : builds own package (dev mode)
- [ ] python setup.py develop : installs own package in an editable way (dev mode)


## Data preprocessing, dataset class and dataloaders (I.Milestone) ##
The original dataset can be downloaded from [here](https://osf.io/nrgx6/). We are using [extract_features.py](https://github.com/neuralinterfacinglab/SingleWordProductionDutch/blob/main/extract_features.py) from [SingleWordProductionDutch](https://github.com/neuralinterfacinglab/SingleWordProductionDutch/tree/28fb2d2db4c3332ba95f831208ffb5dd3dcde223) submodule to create the target spectogram and extract features with channel names.
First [extract_features.py](https://github.com/neuralinterfacinglab/SingleWordProductionDutch/blob/main/extract_features.py) should be run which creates the features folder. It can be run with the following command:
```
python SingleWordProductionDutch/extract_features.py
```
After this a *feature* folder will appear under the *SingleWordProductionDutch* folder. *SpectogramDataset* class is using the feature folder. The dataset class is in [spectogram_dataset.py](/spectogram_dataset.py). It normalizes the given tensors and gives back a windowed feature tensor as input and a windowed spectogram as output. The dimensions of the input is (batch, window, 1143), the scalar 1143 stands for the number of feature channels. The dimensions of the output is (batch, window, 23).
The Dataloader implementation is in the (create_dataloaders.py)[/create_dataloaders.py]. In this file the spectograms and feature tensors are loaded and a json file is made for the normalization. The cross-validation ratio is 80-5-15. 
It loads every participants tensors and split each of them up. It is done by this way, beacuse we do not want test set to be made from only one participants recordings.
The function in this file gives back the dataloaders.
