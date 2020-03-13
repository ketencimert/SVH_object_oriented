# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:46:48 2020

@author: Mert Ketenci
"""
import tensorflow as tf 

def CreateConv2DLayer(num_conv,conv_filter_size, conv_kernel_size, conv_stride, pool_filter_size, 
           pool_stride, conv_padding ='same', use_pool = True, 
           pool_padding = 'same'):
    
    with tf.variable_scope('CNN{}'.format(num_conv)):
        conv2d = tf.keras.layers.Conv2D(filters=conv_filter_size, strides = conv_stride,
                                    kernel_size=conv_kernel_size, padding=conv_padding)
    if use_pool:
        pool = tf.keras.layers.MaxPool2D(pool_size=pool_filter_size, 
                                       strides=pool_stride, padding = pool_padding)
        return [conv2d,pool]
    else:
        return [conv2d]

def Flatten(x):
    return tf.keras.layers.Flatten()(x)

def CreateDenseLayer(num_dense, units, activation = "relu", p_dropout=0.2):
    if activation == 'relu':
        activation = tf.nn.relu
    with tf.variable_scope('D{}'.format(num_dense)):
        dense = tf.keras.layers.Dense(units=units, activation=activation)
        dropout = tf.keras.layers.Dropout(rate=p_dropout)
    return [dense,dropout]

def CreateDigitLayer(num_digit):
    with tf.variable_scope('digit{}'.format(num_digit)):
        digit =tf.keras.layers.Dense(units=11)
    return digit

def CreateLengthLayer(num_digit):
    length = tf.keras.layers.Dense(units=num_digit+2)
    return length