## Spectogram classifier ##
The aim of this subproject is to generate audio signals from spectrograms with the help of a word classifier. This is usefull for the main project as well, because 
this way the original spectrorgram to wav converter script can be replaced.
NOTE:The data folder and its contents will be generated automatically by the scripts.

### Preprocessing ###
To create a classifier compatible dataset from the original SingleWordProductionDutch-iBIDS folder use the [create_audio_dataset](https://github.com/vandahalasi/BRAIN2SPEECH_LesssGoo/blob/main/spec_classifier/create_audio_dataset.ipynb).
This will get every subjects audio recordings and the list of the corresponding spoken words. Those items will be stored in "data\uncut_audios_dir". After this operation the audio files
will be splitten into words and stored in seperate directory. (e.g.: audios saying "door" will be under the \data\words\door)
The problem of this dataset is that there are a lot of classes(around 100 words) and every class has small amount of samples. We can collect more samples manually using the
[audio_recording.ipynb](https://github.com/vandahalasi/BRAIN2SPEECH_LesssGoo/blob/main/spec_classifier/audio_recording.ipynb) notebook. How to use it:
1. Run the corresponding cells then enter the word you will say
2. Press 's' to start the recording.
3. Say the word you entered, you can say that word multiple times, the script will cut the recordings into words. This way you can quickly populate your dataset.
4. Press 'q' to stop the recording.
The script will save the recorded words into the "own_recordings\[entered_word]" directory.

### Training ###
I trained a spectorgram classifier. The audio to spectrogram conversion is also done by the [train.ipynb](https://github.com/vandahalasi/BRAIN2SPEECH_LesssGoo/blob/main/spec_classifier/train.ipynb).
This notebook also contains a lot of visualazation. At the end of this script the trained model is saved into the models directory (created automatically). This way
it can be easily used for inference later on.

### Inferencing ###
To try out the trained model use the (inference.ipynb)(https://github.com/vandahalasi/BRAIN2SPEECH_LesssGoo/blob/main/spec_classifier/inference.ipynb). You can record a word
and the script will print the word you said. Also from the class labels(words) we can generate audio files with the use of the [text_to_speech.ipynb](https://github.com/vandahalasi/BRAIN2SPEECH_LesssGoo/blob/main/spec_classifier/text_to_speech.ipynb)
The generated audio files will be stored in data\generated folder.
