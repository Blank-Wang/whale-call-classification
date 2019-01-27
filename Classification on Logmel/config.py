import numpy as np
import pandas as pd
class Config(object):
    def __init__(self,
                 sampling_rate=22050, audio_duration=1.5, n_classes=16,
                 train_dir="/media/user/Duty/Data_Bank/whale-call-dataset/whale-data-4-fold/development/audio", test_dir="/media/user/Duty/Data_Bank/whale-call-dataset/whale-data-4-fold/evaluation/audio",
                 data_dir='./pickle_files',
                 wave_dir='./pickle_files',
                 model_dir='./model',
                 prediction_dir='./prediction',
                 arch='WaveResnext', pretrain=False,
                 cuda=True, print_freq=10, epochs=50,
                 batch_size=25,
                 momentum=0.9, weight_decay=0.0005,
                 n_folds=5, lr=0.01,
                 n_mels=64, frame_weigth=128, frame_shift=32,
                 debug=False):

        self.labels = ['audio1','audio2','audio3','audio4','audio6','audio7','audio8','audio9','audio10','audio12','audio13','audio15','audio17','audio19','audio22','audio24']

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.audio_length = int(self.sampling_rate * self.audio_duration)
        self.n_classes = n_classes
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.logmel_dir = wave_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.prediction_dir = prediction_dir
        self.arch = arch
        self.pretrain = pretrain
        self.cuda = cuda
        self.print_freq = print_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_folds = n_folds
        self.lr = lr

        self.n_fft = int(frame_weigth / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weigth
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)

        self.debug = debug


if __name__ == "__main__":
    config = Config()
