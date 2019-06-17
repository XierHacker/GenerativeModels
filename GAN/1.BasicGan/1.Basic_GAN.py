import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm
import seaborn

#parameters
MAX_EPOCH=3000
BATCH_SIZE=12
LEARNING_RATE=0.001
UNITS_LAYER1=28         #only one hidden layer
#UNITS_LAYER2=128
#CODE_DIM=200


class RealDistribution():
    def __init__(self,mean=4,variance=0.5):
        self.mean=mean
        self.variance=variance

    def sample(self,N):
        samples=np.random.normal(loc=self.mean,scale=self.variance,size=N)
        samples.sort()
        return samples


#noise input of G
class NoiseDistribution():
    def __init__(self,range):
        self.range=range

    def sample(self,N):
        return np.linspace(start=-self.range,stop=self.range,num=N)+np.random.random(size=N)*0.01


class Basic_GAN():
    def __init__(self,data,noise):
        self.data=data
        self.noise=noise
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.max_epoch = MAX_EPOCH
        self.units_layer1=UNITS_LAYER1
        self.forward()

    #3 layer MLP
    def discriminator(self,input,h_dim,reuse=False):
        #layer1
        with tf.variable_scope(name_or_scope="d1",reuse=reuse):
            weights=tf.get_variable(
                name="weights",
                shape=(input.get_shape[1],h_dim*2),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            bias=tf.get_variable(
                name="bias",
                shape=(h_dim*2,),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            h_1 = tf.nn.relu(tf.matmul(input, weights) + bias)  #shape [batch_size,h_dim*2]

        # layer2
        with tf.variable_scope(name_or_scope="d2", reuse=reuse):
            weights = tf.get_variable(
                name="weights",
                shape=(h_1.get_shape[1], h_dim * 2),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            bias = tf.get_variable(
                name="bias",
                shape=(h_dim * 2,),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            h_2 = tf.nn.relu(tf.matmul(h1, weights) + bias)

        # layer3
        with tf.variable_scope(name_or_scope="d3", reuse=reuse):
            weights = tf.get_variable(
                name="weights",
                shape=(h_2.get_shape[1], 1),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            bias = tf.get_variable(
                name="bias",
                shape=(1,),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            h_3 = tf.nn.relu(tf.matmul(h2, weights) + bias)              #shape [batch_size,1]

        return h3


    #2 layer MLP
    def generator(self,noise,h_dim,reuse=False):
        # layer1
        with tf.variable_scope(name_or_scope="g1", reuse=reuse):
            weights = tf.get_variable(
                name="weights",
                shape=(noise.get_shape[1], h_dim),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            bias = tf.get_variable(
                name="bias",
                shape=(h_dim,),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            h_1 = tf.nn.softplus(tf.matmul(input, weights) + bias)  # shape [batch_size,h_dim]

        # layer2
        with tf.variable_scope(name_or_scope="g2", reuse=reuse):
            weights = tf.get_variable(
                name="weights",
                shape=(h_1.get_shape[1], 1),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            bias = tf.get_variable(
                name="bias",
                shape=(1,),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal()
            )
            h_2 = tf.matmul(h1, weights)+bias      #shape:[batch_size,h_dim]

        return h_2

    def optimizer(self,loss,var_list,initial_learning_rate):
        #decay = 0.95
        #num_decay_steps = 150
        #batch = tf.Variable(0)
        #learning_rate = tf.train.exponential_decay(
        #    initial_learning_rate,
        #    batch,
        #    num_decay_steps,
        #    decay,
        #    staircase=True
        #)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,
            #global_step=batch,
            var_list=var_list
        )
        return optimizer


    def forward(self):
        #pre-Discriminator
        with tf.variable_scope("D_pre"):
            #data placeholder
            self.pre_input_p=tf.placeholder(dtype=tf.float32,shape=(None,1),name="pre_input_p")
            self.pre_labels_p=tf.placeholder(dtype=tf.float32,shape=(None,1),name="pre_labels_p")
            D_pre=self.discriminator(input=self.pre_input_p,h_dim=self.units_layer1,reuse=False)

            #pre-loss
            self.pre_loss=tf.losses.mean_squared_error(labels=self.pre_labels_p,predictions=D_pre)
            self.pre_opt=self.optimizer(loss=self.pre_loss,var_list=None,initial_learning_rate=self.learning_rate)

        #generator
        with tf.variable_scope("generator"):
            #data placeholder
            self.z_p=tf.placeholder(dtype=tf.float32,shape=(None,1),name="z_p")
            self.G=self.generator(noise=self.z_p,h_dim=self.units_layer1,reuse=False)

        #discriminator
        with tf.variable_scope("discriminator"):
            #data palceholder
            self.x_p=tf.placeholder(dtype=tf.float32,shape=(None,1),name="x_p")
            self.D1=self.discriminator(input=self.x_p,h_dim=self.units_layer1,reuse=False)
            self.D2=self.discriminator(input=self.G,h_dim=self.units_layer1,reuse=True)




    def fit(self):
        pass



if __name__=="__main__":
    realdist=RealDistribution()
    real_samples=realdist.sample(10000)
    print(real_samples)
    #print("real_samples.shape",real_samples.shape)
    #plt.plot([i for i in range(-5000,5000)],real_samples)
    #plt.show()