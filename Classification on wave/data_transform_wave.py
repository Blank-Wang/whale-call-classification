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
    """Save variable into a pickle file   """
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
    pool = Pool(14)
    pool.map(tsfm_wave, df.iterrows())

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

if __name__ == '__main__':
    # make_dirs()
    config = Config(sampling_rate=22050,n_classes=16)

    get_wavelist()
    if os.path.exists(config.data_dir):
        shutil.rmtree(config.data_dir)
    os.makedirs(config.data_dir)
    wav_to_pickle('wavelist.csv')
