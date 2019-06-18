import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import dcgan

epoch=1
CHECK_POINTS_PATH="./checkpoints/dcgan-"+str(epoch)

def generate(noise):
    #generator model
    G=dcgan.Generator(
        fc_shapes=(7,7,256),
        num_conv=3,
        filters=(128,64,1),
        kernel_sizes=(5,5,5),
        strides=(1,2,2),
        activations=(tf.nn.leaky_relu,tf.nn.leaky_relu,tf.nn.tanh)
    )

    checkpoint = tf.train.Checkpoint(G=G)
    checkpoint.restore(CHECK_POINTS_PATH)

    #generate
    fake_images=G(noise,training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(fake_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(fake_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

if __name__=="__main__":
    noise=tf.random.normal(shape=(16,100))
    print("noise:\n",noise)
    generate(noise=noise)
