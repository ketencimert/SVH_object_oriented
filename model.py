# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:46:48 2020

@author: Mert Ketenci
"""
import tensorflow as tf
from compute_utils import *

class Model(tf.keras.Model):
    def __init__(self, conv_params,dense_params,digit_params):
        super().__init__()
        
        (conv_filter_size, conv_kernel_size, conv_stride, pool_filter_size,
         pool_stride,conv_padding,use_pool,pool_padding)=conv_params
        
        units, activation, p_dropout=dense_params
        num_digit = digit_params   

        self.ConvLayers=[]
        for i in range(len(conv_filter_size)):
            for layer in CreateConv2DLayer(i,conv_filter_size[i], conv_kernel_size[i], conv_stride[i], 
                                     pool_filter_size[i], pool_stride[i], conv_padding[i], use_pool[i],
                                     pool_padding[i]):  
                self.ConvLayers.append(layer)        
        
        self.DenseLayers = []
        for j in range(len(units)):
            for layer in CreateDenseLayer(j, units[j], activation[j], p_dropout[j]):
                self.DenseLayers.append(layer)
           
        self.DigitLayers = []
        for k in range(num_digit):
            self.DigitLayers.append(CreateDigitLayer(k))
        
        self.LengthLayer = CreateLengthLayer(num_digit)
        
    def forward(self,x,LabelLength,LabelDigits):

        for ConvLayer in self.ConvLayers:
            x = ConvLayer(x)
        
        x = Flatten(x)
        
        for DenseLayer in self.DenseLayers:
            x = DenseLayer(x)
        
        digits =[]
        for DigitLayer in self.DigitLayers:
            digits.append(DigitLayer(x))
        LogitDigits = tf.stack(digits, axis=1)
        
        LogitLength = self.LengthLayer(x)
        
        with tf.name_scope("loss"):
            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=LabelLength, logits=LogitLength))
            for i in range(LogitDigits.shape[1]):
                total_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=LabelDigits[:, i], 
                                                                                     logits=LogitDigits[:, i, :]))
        return total_loss


