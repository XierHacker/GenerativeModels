import sys
sys.path.append("..")

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from util import data
import auto_encoder

MAX_EPOCH=30
BATCH_SIZE=100
LEARNING_RATE=0.001


def train(X):
    #plaec holder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,784),name="X_p")

    #regularizer
    regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)

    model=auto_encoder.AutoEncoder()

    #encode
    code=model.encoder(X_p,regularizer,"encoder",False)
    #decode
    generate=model.decoder(code,regularizer,"decoder",False)

    #loss
    loss=tf.losses.mean_squared_error(
        labels=tf.reshape(X_p,[-1]),predictions=tf.reshape(generate,[-1])
    )

    optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    init_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()

    with tf.Session() as sess:
        print("Training Start")
        sess.run(init_op)  # initialize all variables
        sess.run(init_local_op)
        train_Size = X.shape[0]
        for epoch in range(1, MAX_EPOCH + 1):
            losses=[]
            print("Epoch:", epoch)
            start_time = time.time()  # time evaluation

            # mini batch
            for i in range(0, (train_Size // BATCH_SIZE)):
                _,loss_=sess.run(
                    fetches=(optimizer,loss),
                    feed_dict={X_p:X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]}
                )
                losses.append(loss_)

            print("loss:",sum(losses)/len(losses))

            #store the first pictre
            gen_pic=sess.run(
                fetches=generate,
                feed_dict={X_p:X}
            )
            #print("gen_pic:\n",gen_pic)
            plt.imshow(np.reshape(X[3],newshape=[28,28]))
            plt.imshow(np.reshape(gen_pic[3],newshape=[28,28]))
            plt.show()




if __name__=="__main__":
    print("Load Data....")
    X,y_train,X_test=data.load_mnist(reshape=False)
    X,X_test=data.standard(X_train=X,X_test=X_test)
    print(X.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(X[0])

    print("Run Model....")
    train(X)

