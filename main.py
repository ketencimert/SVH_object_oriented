# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:10:32 2020

@author: Mert Ketenci
"""
import argparse
import numpy as np
from preprocess import preprocess
import tensorflow as tf
from tqdm import tqdm

from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Wiki category generator')
    parser.add_argument('--file_name', default = 'trainpkl.gz')
    parser.add_argument('-split', default = 0.2)
    parser.add_argument('-batch_size', default = 32)
    parser.add_argument('-epoches', default = 30)
    parser.add_argument('-learning_rate', default = 1e-5)
    parser.add_argument('-conv_filter_size', default = [10,20])
    parser.add_argument('-conv_kernel_size', default = [[2, 2],[2, 2]])
    parser.add_argument('-conv_stride', default = [[2, 2],[2, 2]])
    parser.add_argument('-use_pool', default = [True,False])
    parser.add_argument('-conv_padding', default = ['same','same'])
    parser.add_argument('-pool_filter_size', default = [[2, 2],[2, 2]])
    parser.add_argument('-pool_stride', default = [[2, 2],[2, 2]])
    parser.add_argument('-pool_padding', default =['same','same'])
    parser.add_argument('-dense_units', default =[100,100])
    parser.add_argument('-dense_activation', default =['relu','relu'])
    parser.add_argument('-p_dropout', default = [0.1,0.1])
    parser.add_argument('-num_digit', default = 5)
    args = parser.parse_args()
    
    X_train, X_val, yd_train, yl_train = preprocess(args.file_name,args.split)
    
    num_train = X_train.shape[0]
    num_batch = num_train//args.batch_size

    conv_params = (args.conv_filter_size, args.conv_kernel_size, args.conv_stride, 
                   args.pool_filter_size,args.pool_stride, 
                   args.conv_padding, args.use_pool, args.pool_padding)
    
    dense_params = (args.dense_units, args.dense_activation, args.p_dropout)    
    
    model = Model(conv_params,dense_params,args.num_digit)
    
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=(None,54,54,3))
        yl = tf.placeholder(tf.int64, shape=(None,None,))
        yd =  tf.placeholder(tf.int64, shape=(None,None,None))
    
    loss = model.forward(X,yl,yd)
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    
    iter_total = 0
    loss_history =[]
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for e in range(args.epoches):
            for i in tqdm(range(num_batch)):
                iter_total += 1
                choice=np.random.choice(num_train, size=args.batch_size, replace=False)
                batch_x, batch_digits, batch_length = X_train[choice], yd_train[choice], yl_train[choice]
                _, loss_tf = sess.run([train_step,loss],
                                                                          feed_dict={X:batch_x , 
                                                                                     yl:batch_length,
                                                                                     yd:batch_digits})
                loss_history.append(loss_tf)
            print('Epoch loss:{}\n'.format(loss_history[-1]))