#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:28:27 2021

@author: limbu
"""

#使用Log和MinMax缩放为英语和瑞典语音频片段创建3d阵列矩阵
#50000个文件需要3.12小时

import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")

d = 1
flag = 1
count = 0
samples = 1500

En = np.zeros((samples,128,200))
maxE = np.zeros(samples + 1)

Sv = np.zeros((samples,128,200))
maxS = np.zeros(samples + 1)

start = time.time()
while(d<(samples + 1)):
    
    yE, srE = librosa.load("\content\drive\MyDrive\S2S Parallel data\zh\di"+str(d)+"ju.wav")
    Se = librosa.feature.melspectrogram(y=yE, sr=srE, n_mels=128,fmax=srE/2)
    colE = Se.shape[1]
    maxE[d-1] = colE

    yS, srS = librosa.load("\content\drive\MyDrive\S2S Parallel data\tibet\di"+str(d)+"ju.wav")
    Ss = librosa.feature.melspectrogram(y=yS, sr=srS, n_mels=128,fmax=srS/2)
    colS = Ss.shape[1]
    maxS[d-1] = colS
    
    #Ignoring all speech files that take more than 200 timesteps (to filter out long sentences)
    #Break loop for current iteration and start over with new d value.
    
    if(colE >= 100 or colS >= 100):
        count = count + 1
        d = d+1
        continue
    else:
        e = 0
        
        #Log Scaling
        Se = np.log((Se+10e-8))
        Ss = np.log((Ss+10e-8))

        while(e<128):
            En[flag-1][e][:colE] = Se[e]
            Sv[flag-1][e][:colS] = Ss[e]
            e = e+1
        En[flag-1] = (En[flag-1] - np.min(En[flag-1]))/(np.max(En[flag-1])-np.min(En[flag-1]))
        Sv[flag-1] = (Sv[flag-1] - np.min(Sv[flag-1]))/(np.max(Sv[flag-1])-np.min(Sv[flag-1]))
        if (d%100 == 0):
            print(d)

        d = d+1
        flag = flag+1
        
#为剩余值创建新变量

Res = samples - count
en = np.zeros((Res,128,200))
sv = np.zeros((Res,128,200))
en = En[0:Res]
sv = Sv[0:Res]
    
end = time.time()
print(end - start)

#转换数据以获得LSTM所需的格式

S = np.zeros((Res,200,128))
for i in range(0,Res):
    #row = s.shape[2]
    S[i] = sv[i].T 
E = np.zeros((Res,200,128))
for j in range(0,Res):
    E[j] = en[j].T

print('Total Data obtained : ' + str(Res))


#将数据另存为pickle，以便在Kaggle内核或Google Cloud GPU中运行计算
import pickle
import time
start = time.time()
k = open('Eng'+str(Res)+'.pckl', 'wb')
pickle.dump(E, k,protocol = 4)
k.close()

r = open('Swe'+str(Res)+'.pckl', 'wb')
pickle.dump(S, r,protocol = 4)
r.close()

end = time.time()
print(end - start)
