'''
Created on Jul 13, 2018

@author: Dezhi Wang
'''
import csv
from itertools import groupby
import pandas as pd
import os


df1 = pd.read_csv('./fold1_train.txt', sep='\t', names=['fname','label'])
df2 = pd.read_csv('./fold2_train.txt', sep='\t', names=['fname','label'])
df3 = pd.read_csv('./fold3_train.txt', sep='\t', names=['fname','label'])
df4 = pd.read_csv('./fold4_train.txt', sep='\t', names=['fname','label'])

valid = pd.read_csv('./evaluate.txt', sep='\t', names=['fname','label'])

# wavelist=pd.concat([df1,df2,df3,df4,valid],ignore_index=True)
train=pd.concat([df1,df2,df3,df4],ignore_index=True)
print('train csv file shape is:', train.shape)
train.drop_duplicates(subset=['fname'],keep='first',inplace=True)
valid.drop_duplicates(subset=['fname'],keep='first',inplace=True)
print('train csv file shape is:', train.shape)
print('valid csv file shape is:', valid.shape)

# Test the wav files
new_key=[]
new_event=[]
wav_train= "/media/user/Duty/Data_Bank/whale-call-dataset/whale-data-4-fold/development/audio"
wav_valid= "/media/user/Duty/Data_Bank/whale-call-dataset/whale-data-4-fold/evaluation/audio"

names = [na for na in os.listdir(wav_train) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in train.iterrows():
        
        fe_path = os.path.join(wav_train, row['fname'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['fname'])     
            train.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
train.to_csv("./whale_train.csv", index=False, sep='\t', columns=['fname','label'])


names = [na for na in os.listdir(wav_valid) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in valid.iterrows():
        
        fe_path = os.path.join(wav_valid, row['fname'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['fname'])     
            valid.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
valid.to_csv("./whale_valid.csv", index=False, sep='\t', columns=['fname','label'])

print('Conversion Completed')

        
        
        