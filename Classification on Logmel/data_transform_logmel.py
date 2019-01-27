#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# from util import *
import os
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from config import *
import librosa
import shutil
import pickle
def save_data(filename, data):
    """Save variable into a pickle file """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))
def get_wavelist():
    train_dir = config.train_dir
    test_dir = config.test_dir
    waves_train = sorted(os.listdir(train_dir))
    waves_test = sorted(os.listdir(test_dir))
    print(len(waves_train)+len(waves_test))
    df_train = pd.DataFrame({'fname': waves_train})
    df_train['train0/test1'] = pd.DataFrame(0 for i in range(len(waves_train)))

    df_test = pd.DataFrame({'fname': waves_test})
    df_test['train0/test1'] = pd.DataFrame(1 for i in range(len(waves_test)))

    df = df_train.append(df_test)
    df.set_index('fname', inplace=True)
    df.to_csv('./wavelist.csv')

def wav_to_pickle(wavelist):

    df = pd.read_csv(wavelist, index_col=None)
    # print(df)
    pool = Pool(10)
    pool.map(tsfm_wave, df.iterrows())


def wav_to_logmel(wavelist):

    df = pd.read_csv(wavelist)
    # print(df)
    pool = Pool(14)
    pool.map(tsfm_logmel, df.iterrows())


def wav_to_mfcc(wavelist):

    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_mfcc, df.iterrows())


def tsfm_wave(row):
    item = row[1]
    
    sr=config.sampling_rate #resample to required samplingrate
    
    if item['train0/test1'] == 0:
        file_path = os.path.join(config.train_dir, item['fname'])
    elif item['train0/test1'] == 1:
        file_path = os.path.join(config.test_dir, item['fname'])

    print(row[0], file_path)
    data, _ = librosa.core.load(file_path, sr=sr, res_type='kaiser_best')
    p_name = os.path.join(config.data_dir, os.path.splitext(item['fname'])[0] + '.pkl')
    save_data(p_name, data)

def tsfm_logmel(row):
    item = row[1]
    p_name = os.path.join(config.data_dir, os.path.splitext(item['fname'])[0] + '.pkl')
    
    if not os.path.exists(p_name):
        print(str(row[0])+p_name)
        if item['train0/test1'] == 0:
            file_path = os.path.join(config.train_dir, item['fname'])
        elif item['train0/test1'] == 1:
            file_path = os.path.join(config.test_dir, item['fname'])
        data, sr = librosa.load(file_path, sr=config.sampling_rate)

        if len(data) == 0:
            print("empty file:", file_path)
            logmel = np.zeros((config.n_mels, int(config.audio_length/config.hop_length)))
            feats = np.stack((logmel, logmel, logmel))
        else:
            melspec = librosa.feature.melspectrogram(data, sr,
                                                     n_fft=config.n_fft, hop_length=config.hop_length,
                                                     n_mels=config.n_mels)

            logmel = librosa.core.power_to_db(melspec)
            
            delta = librosa.feature.delta(logmel, order=1)
            accelerate = librosa.feature.delta(logmel, order=2)
#             delta = librosa.feature.delta(logmel, order=1, mode='nearest')
#             accelerate = librosa.feature.delta(logmel, order=2, mode='nearest')
            feats = np.stack((logmel, delta, accelerate)) #(3, 64, xx)
#             print(feats.shape)
        save_data(p_name, feats)
        

def tsfm_mfcc(row):
    item = row[1]
    
    p_name = os.path.join(config.data_dir, os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        # print(p_name)
        if item['train0/test1'] == 0:
            file_path = os.path.join('../audio_train/', item['fname'])
        elif item['train0/test1'] == 1:
            file_path = os.path.join('../audio_test/', item['fname'])
        data, sr = librosa.load(file_path, config.sampling_rate)

        if len(data) == 0:
            print("empty file:", file_path)
            mfcc = np.zeros((config.n_mels, int(config.audio_length/config.hop_length)))
            feats = np.stack((mfcc, mfcc, mfcc))
        else:
            mfcc = librosa.feature.mfcc(data, sr,
                                        n_fft=config.n_fft,
                                        hop_length=config.hop_length,
                                        n_mfcc=config.n_mels)
            delta = librosa.feature.delta(mfcc)
            accelerate = librosa.feature.delta(mfcc, order=2)
            feats = np.stack((mfcc, delta, accelerate)) #(3, 64, xx)        
        save_data(p_name, feats)

if __name__ == '__main__':
#     config = Config(sampling_rate=22050,n_classes=16)
    config = Config(sampling_rate=22050, n_mels=128, frame_weigth=128, frame_shift=10)
    get_wavelist()
    if os.path.exists(config.data_dir):
        shutil.rmtree(config.data_dir)
    os.makedirs(config.data_dir)
#     wav_to_pickle('wavelist.csv')
    wav_to_logmel('wavelist.csv')
    # wav_to_mfcc('wavelist.csv')