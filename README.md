### Title
Large-scale Whale-call Classification by Transfer Learning on Multi-scale Waveforms and Time-frequency Features
Authors: Dezhi Wang, Lilun Zhang, Changchun Bao, Yongxian Wang and Kele Xu
College of Meteorology and Oceanography, National University of Defense Technology, China

https://github.com/Blank-Wang/whale-call-classification

### Dataset
The experimental dataset comes from the Whale FM website(available from: https://whale.fm and https://github.com/zooniverse /WhaleFM/blob/master/csv/whale_fm_anon_04-03-2015_assets.csv.). It is a citizen science project from Zooniverse and Scientific American.


### Run:

Including two tasks:

1, classification on logmel

run data_transform_logmel.py to pre-process the whale-call data

run train_on_logmel.py to do the classification

run make_predictions_4.py to evaluate the system performance


2, classificaiton on waveforms

run data_transform_wave.py to pre-process the whale-call data

run train_on_wave.py to do the classification

run make_predictions_3.py to evaluate the system performance

### Requirments:

python 3.6

pytorch 0.4.0

cuda 9.1

librosa 0.5.1

torchvision 0.2.1

