# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:10:32 2020

@author: Mert Ketenci
"""
import gzip
import numpy as np
import os
import six.moves.cPickle as pickle

def preprocess(file_name,split):
    
    f = gzip.open(os.path.join(os.getcwd(), file_name), 'rb')
    SVH = pickle.load(f)
    f.close()

    digits=[]
    lengths=[]
    for i in range(len(SVH["labels"])):
        digits.append([int(x) for x in SVH["labels"][i]])
        lengths.append(len(SVH["labels"][i]))
        
    N = len(lengths)                
    digit_matrix=np.zeros((N,6,11), dtype=int)
    for i in range(N):
        for j in range(len(digits[i])):
            digit_matrix[i][j][digits[i][j]]=1
            for j in range(len(digits[i]),6):
                digit_matrix[i][j][10]=1
            
    length_matrix = np.zeros((N, 7),dtype=int)
    length_matrix[np.arange(N,), lengths] = 1
    
    N_train = int(N*(1-split))
    images = SVH["images"]

    X_train = np.array(images[:N_train])
    X_val = np.array(images[N_train:N])
    mean_image = np.mean(X_train, axis=0)
    X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
    X_val = X_val.astype(np.float32) - mean_image
    
    X_train = X_train.reshape([-1,54,54,3])
    X_val = X_val.reshape([-1,54,54,3])
    
    yd_train = digit_matrix[:N_train]
    yl_train = length_matrix[:N_train]
    
    return X_train, X_val, yd_train, yl_train