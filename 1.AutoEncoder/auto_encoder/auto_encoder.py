'''
two-layer mlp
'''
import sys
sys.path.append("..")

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from util import data

#config
#os.environ['CUDA_VISIBLE_DEVICES']='2'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

#parameters
IMAGE_SIZE=28
IMAGE_FLATTEN_DIM=IMAGE_SIZE*IMAGE_SIZE
UNITS_LAYER1=256
UNITS_LAYER2=128
CODE_DIM=2

class AutoEncoder():
    def __init__(self):
        self.image_size=IMAGE_SIZE
        self.image_flatten_dim = IMAGE_FLATTEN_DIM
        self.units_layer1 = UNITS_LAYER1
        self.units_layer2 = UNITS_LAYER2
        self.code_dim=CODE_DIM

    #pytorch-like API,fully connected layer
    def linear(self,X,units,activation,regularizer,name,reuse):
        # layer 1
        logits = tf.layers.dense(
            inputs=X,units=units,activation=activation,use_bias=True,
            kernel_initializer=None, bias_initializer=tf.initializers.constant(),
            kernel_regularizer=regularizer, bias_regularizer=regularizer,
            trainable=True, name=name,reuse=reuse
        )
        return logits

    #encoder
    def encoder(self,X,regularizer,name_scope="encoder",reuse=False):
        with tf.variable_scope(name_or_scope=name_scope,reuse=reuse):
            #layer 1
            logits_1=self.linear(X,self.units_layer1,tf.nn.relu,regularizer,"encoder_logits_1",reuse)
            #layer 2
            logits_2 = self.linear(logits_1, self.units_layer2, tf.nn.relu, regularizer, "encoder_logits_2", reuse)
            #output(encoding layer),No activation
            code=self.linear(logits_2, self.code_dim, None, regularizer, "code", reuse)
            return code


    #decoder
    def decoder(self,code,regularizer,name_scope="decoder",reuse=False):
        with tf.variable_scope(name_or_scope=name_scope,reuse=reuse):
            # layer 1
            logits_1 = self.linear(code, self.units_layer2, tf.nn.relu, regularizer, "decoder_logits_1", reuse)
            # layer 2
            logits_2 = self.linear(logits_1, self.units_layer1, tf.nn.relu, regularizer, "decoder_logits_2", reuse)
            # output(encoding layer),No activation
            X = self.linear(logits_2, self.image_flatten_dim, None, regularizer, "X", reuse)
            return X




