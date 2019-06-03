import os, sys, json
import numpy as np
import torch
import torch.utils.data
from sklearn.linear_model import LogisticRegression

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'

estimation_X_f1 = [1,2,3]
estimation_X_f2 = [2,4,6]
estimation_y = [1,1,0]
estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial')

X = []
y = []
for i in range(len(estimation_X_f1)):
    X_elem = [estimation_X_f1[i], estimation_X_f2[i]]    
    X.append(X_elem)    
    y.append(estimation_y[i])

estimator.fit(X,y)

pred = estimator.predict([[3,5.2]])
print(pred[0])


from scipy import stats
ds = stats.describe([5,5,5,6,9,20,20,45,200])
print(ds)
cnt = ds[0]
min = ds[1][0]
max = ds[1][1]
mean = ds[2]
variance = ds[3]
skewness = ds[4]
kurtosis = ds[5]
print(kurtosis)

vec = [5,5,5,6,9,20,20,45,200]
len = len(vec)
i_0 = vec[0]
i_25 = vec[int(len/4)]
i_50 = vec[int(len/2)]
i_75 = vec[int(3*len/4)]
i_100 = vec[-1]
print(i_75)
