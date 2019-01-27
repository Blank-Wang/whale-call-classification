'''
Created on Jul 13, 2018

@author: Dezhi Wang
'''
import csv
from itertools import groupby
import pandas as pd
import os
from numpy import asmatrix

labels = ['audio10','audio12','audio13','audio15','audio17','audio19', 'audio1', 'audio22', 'audio24', 'audio2', 'audio3', 'audio4','audio6', 'audio7', 'audio8','audio9']
# df1 = pd.read_csv('./prediction/Ensemble_predictions.csv', sep='\t', names=['fname','probs'])
df1 = pd.read_csv('./prediction/Ensemble_predictions.csv', header=None)
dfp=df1.values

class_label=[]
prob_values=[]
file_names=[]
for ind, row in enumerate(dfp):
    file=row[0]
    file_names.append(file)
    classname=file.split('_')[1]
    class_label.append(classname)
    probability=row[1:]
    prob_values.append(probability)
# df1['classname']=class_label
# df1.insert(0, 'classname', class_label)

dfu=pd.DataFrame(file_names,columns=['filename'])
dfu.insert(0, 'classname', class_label)
dfu.insert(2, 'probs', prob_values)

print(dfu.shape)

aaa=dfu['probs'].groupby(dfu['classname'])

namelist=[]
probmatrix=[]
for gname, gdata in aaa:
    print(gname)
#     print(gdata)
    print(gdata.mean())
    namelist.append(gname)
    probmatrix.append(gdata.mean())
    
smatrix=pd.DataFrame(probmatrix,columns=labels)
smatrix.insert(0,'classes', namelist)
smatrix.to_csv('smatrix.csv', sep='\t')

ssmatrix=smatrix.sort_index(axis=1)
ssmatrix.to_csv('ssmatrix.csv', sep='\t')

print('good')    
    


        
        
        