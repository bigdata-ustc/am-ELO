# -*- coding: utf-8 -*-


import numpy as np
import torch
import random
import os

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def sigmoid(X):
    if X>0:
        return 1/(1+np.exp(-X))
    else:
        return np.exp(X)/(1+np.exp(X))
    
def get_pred(R,A,i,j,k):   
    return sigmoid(A[k]*(R[i]-R[j]))


def AUC(X,Y):
    s = 0
    total = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i]>X[j]:
                total += 1
                if Y[i]>Y[j]:
                    s += 1
    return s/total