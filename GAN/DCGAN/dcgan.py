import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

class Generator(tf.keras.Model):
    def __init__(self,fc_shapes,num_conv,filters,kernel_sizes,strides,activations):
        '''

        '''
        super(Generator,self).__init__()
        self.fc_shapes=fc_shapes
        #linear transform
        self.fc=tf.keras.layers.Dense(
            units=fc_shapes[0]*fc_shapes[1]*fc_shapes[2],
            activation=tf.nn.leaky_relu,
            use_bias=False
            )

        #deconvlutions
        self.de_convs=[]
        for i in range(num_conv):
            de_conv=tf.keras.layers.Conv2DTranspose(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                strides=strides[i],
                activation=activations[i],
                padding="same"
                )
            self.de_convs.append(de_conv)
        #batch normalization
        self.BN=tf.keras.layers.BatchNormalization()
        

    def __call__(self, inputs,training=True):
        #fc
        x=self.fc(inputs)
        # x=self.BN(x,training=training)
        #reshape
        x=tf.reshape(tensor=x,shape=(-1,self.fc_shapes[0],self.fc_shapes[1],self.fc_shapes[2]))
        #print("x:\n",x)
        # print("x.shape:",x.shape)
        #
        for de_conv in self.de_convs:
            x=de_conv(x)
            # x=self.BN(x,training=training)
            # print("x.shape:",x.shape)
        return x
        

class Discriminator(tf.keras.Model):
    def __init__(self,num_conv,filters,kernel_sizes,strides,activations):
        super(Discriminator,self).__init__()
        self.convs=[]
        self.dropouts=[]
        for i in range(num_conv):
            conv=tf.keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                strides=strides[i],
                activation=activations[i]
                )
            dropout=tf.keras.layers.Dropout(rate=0.3)
            self.convs.append(conv)
            self.dropouts.append(dropout)
        self.flatten=tf.keras.layers.Flatten()
        self.fc=tf.keras.layers.Dense(units=1)


    def __call__(self, inputs,training=True):
        for i in range(len(self.convs)):
            inputs=self.convs[i](inputs)
            inputs=self.dropouts[i](inputs,training=training)
        inputs=self.flatten(inputs)
        outputs=self.fc(inputs)
        return outputs
            

if __name__=="__main__":
    g=Generator(
        fc_shapes=(7,7,256),
        num_conv=3,
        filters=(128,64,1),
        kernel_sizes=(5,5,5),
        strides=(1,2,2),
        activations=(tf.nn.leaky_relu,tf.nn.leaky_relu,tf.nn.tanh)
        )
    
    noise=tf.random.normal(shape=(1,100))
    print("noise:\n",noise)

    result=g(noise,True).numpy()
    print("result:",result[0])

    d=Discriminator(
        num_conv=2,
        filters=(64,128),
        kernel_sizes=(5,5),
        strides=(2,2),
        activations=(tf.nn.leaky_relu,tf.nn.leaky_relu)
        )
    
    outputs=d(result)
    print("outputs:\n",outputs)

    plt.imshow(X=result[0,:,:,0])
    plt.show()

