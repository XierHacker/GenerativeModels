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
os.environ['CUDA_VISIBLE_DEVICES']='2'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

#parameters
MAX_EPOCH=10
BATCH_SIZE=20
LEARNING_RATE=0.001
IMAGE_SIZE=28
IMAGE_FLATTEN_DIM=784
UNITS_LAYER1=128
UNITS_LAYER2=64
CODE_DIM=30

class AutoEncoder():
    def __init__(self):
        self.graph=tf.Graph()
        self.session=tf.Session(graph=self.graph,config=config)
        self.max_epoch=MAX_EPOCH
        self.batch_size=BATCH_SIZE
        self.learning_rate=LEARNING_RATE
        self.image_size=IMAGE_SIZE
        self.image_flatten_dim = IMAGE_FLATTEN_DIM
        self.units_layer1 = UNITS_LAYER1
        self.units_layer2 = UNITS_LAYER2
        self.code_dim=CODE_DIM

    #encoder
    def encoder(self,X,name_scope="encoder",reuse=False):
        with self.graph.as_default():
            with tf.variable_scope(name_or_scope=name_scope,reuse=False):
                #layer 1
                with tf.variable_scope(name_or_scope="layer_1",reuse=reuse):
                    weight=tf.get_variable(
                        name="weight",
                        shape=(self.image_flatten_dim,self.units_layer1),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias=tf.get_variable(
                        name="bias",
                        shape=(self.units_layer1,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_1=tf.nn.relu(tf.matmul(X,weight)+bias)

                #layer 2
                with tf.variable_scope(name_or_scope="layer_2",reuse=reuse):
                    weight=tf.get_variable(
                        name="weight",
                        shape=(self.units_layer1,self.units_layer2),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias=tf.get_variable(
                        name="bias",
                        shape=(self.units_layer2,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_2=tf.nn.relu(tf.matmul(h_1,weight)+bias)  #[batch_size,code_dim]

                # layer 3
                with tf.variable_scope(name_or_scope="layer_3", reuse=reuse):
                    weight = tf.get_variable(
                        name="weight",
                        shape=(self.units_layer2, self.code_dim),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias = tf.get_variable(
                        name="bias",
                        shape=(self.code_dim,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_3 = tf.nn.relu(tf.matmul(h_2, weight) + bias)  # [batch_size,code_dim]
                    return h_3


    #decoder
    def decoder(self,code,name_scope="decoder",reuse=False):
        with self.graph.as_default():
            with tf.variable_scope(name_or_scope=name_scope,reuse=False):
                #layer1
                with tf.variable_scope("layer_1",reuse=reuse):
                    weight = tf.get_variable(
                        name="weight",
                        shape=(self.code_dim, self.units_layer2),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias = tf.get_variable(
                        name="bias",
                        shape=(self.units_layer2,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_1 = tf.nn.relu(tf.matmul(code, weight) + bias)

                # layer 2
                with tf.variable_scope(name_or_scope="layer_2", reuse=reuse):
                    weight = tf.get_variable(
                        name="weight",
                        shape=(self.units_layer2, self.units_layer1),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias = tf.get_variable(
                        name="bias",
                        shape=(self.units_layer1,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_2 = tf.nn.relu(tf.matmul(h_1, weight) + bias)  # [batch_size,image_flatten_dim]

                # layer 2
                with tf.variable_scope(name_or_scope="layer_3", reuse=reuse):
                    weight = tf.get_variable(
                        name="weight",
                        shape=(self.units_layer1, self.image_flatten_dim),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    bias = tf.get_variable(
                        name="bias",
                        shape=(self.image_flatten_dim,),
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal()
                    )
                    h_3 = tf.nn.relu(tf.matmul(h_2, weight) + bias)  # [batch_size,image_flatten_dim]
                    return h_3


    def fit(self,X_train):
        with self.graph.as_default():
            #plaec holder
            self.X_p=tf.placeholder(dtype=tf.float32,shape=(None,784),name="X_p")

            code=self.encoder(X=self.X_p)
            generate=self.decoder(code=code)

            #loss
            self.loss=tf.reduce_mean(tf.abs(self.X_p-generate))

            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()
            self.init_local_op = tf.local_variables_initializer()

        with self.session as sess:
            print("Training Start")
            sess.run(self.init_op)  # initialize all variables
            sess.run(self.init_local_op)

            train_Size = X_train.shape[0]

            for epoch in range(1, self.max_epoch + 1):
                losses=[]
                print("Epoch:", epoch)
                start_time = time.time()  # time evaluation

                # mini batch
                for i in range(0, (train_Size // self.batch_size)):
                    _,loss=sess.run(
                        fetches=(self.optimizer,self.loss),
                        feed_dict={self.X_p:X_train[i*self.batch_size:(i+1)*self.batch_size]}
                    )
                    losses.append(loss)

                print("loss:",sum(losses)/len(losses))

                #store the first pictre
                gen_pic=sess.run(
                    fetches=generate,
                    feed_dict={self.X_p:X_train[:10]}
                )
                print("gen_pic:\n",gen_pic)
                plt.imshow(np.reshape(gen_pic[1],newshape=[28,28]))
                plt.show()



    def pred(self):
        pass


if __name__=="__main__":
    print("Load Data....")
    X_train,y_train,X_test=data.load_mnist(reshape=False)
    X_train=X_train/256
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)

    print("Run Model....")
    model=AutoEncoder()
    model.fit(X_train=X_train)
