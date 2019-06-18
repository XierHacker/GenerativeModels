import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dcgan

#define strategy
strategy=tf.distribute.MirroredStrategy()
print("num devices:",strategy.num_replicas_in_sync)

#parameters
BATCH_SIZE_PER_REPLICA=128
BATCH_SIZE=BATCH_SIZE_PER_REPLICA*strategy.num_replicas_in_sync
print("batch_size_per_replica:",BATCH_SIZE_PER_REPLICA)
print("batch_size:",BATCH_SIZE)

CLASS_NUM=10
NOISE_DIM=100
EPOCHS=50

#model save path
CHECK_POINTS_PATH="./checkpoints/dcgan"

print(tf.__version__)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()

#expand dim to use convlution 2D
x_train=np.expand_dims(a=x_train,axis=-1)
x_train=(x_train-127.5)/127.5
x_train=x_train.astype(np.float32)
print("train sample size:",len(x_train))
print("x_train.dtype:",x_train.dtype)
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(buffer_size=len(x_train)).batch(BATCH_SIZE)
# for records in train_dataset:
#     print("records:\n",records[0])
#     print("records:\n",records[1])
    

#x_test=np.expand_dims(a=x_test,axis=-1)/np.float32(255)
#test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(BATCH_SIZE)

# plt.imshow(X=x_train[1,:,:,0])
# plt.show()

# print("label:",y_train[1])

def train():
    #trans dataset to distribute dataset
    with strategy.scope():
        train_dist_dataset=strategy.experimental_distribute_dataset(train_dataset)
        #test_dist_dataset=strategy.experimental_distribute_dataset(test_dataset)

    #define loss
    with strategy.scope():
        BinaryEntropy=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
            )
        #generator loss
        def g_loss(fake_logits):
            '''
            '''
            losses=BinaryEntropy(tf.ones_like(fake_logits),fake_logits)
            loss=tf.nn.compute_average_loss(per_example_loss=losses,global_batch_size=BATCH_SIZE)
            return loss

        #discriminator loss
        def d_loss(real_logits,fake_logits):
            '''
            '''
            real_losses=BinaryEntropy(tf.ones_like(real_logits),real_logits)
            fake_losses=BinaryEntropy(tf.zeros_like(fake_logits),fake_logits)
            losses=real_losses+fake_losses
            loss=tf.nn.compute_average_loss(per_example_loss=losses,global_batch_size=BATCH_SIZE)
            return loss

    #define metrics
    # with strategy.scope():
    #     train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name="train_loss")

    # model and optimizer must be created under `strategy.scope`.
    with strategy.scope():
        #generator model
        G=dcgan.Generator(
            fc_shapes=(7,7,256),
            num_conv=3,
            filters=(128,64,1),
            kernel_sizes=(5,5,5),
            strides=(1,2,2),
            activations=(tf.nn.leaky_relu,tf.nn.leaky_relu,tf.nn.tanh)
        )
        G_optimizer=tf.keras.optimizers.Adam(1e-4)

        #discriminator model
        D=dcgan.Discriminator(
            num_conv=2,
            filters=(64,128),
            kernel_sizes=(5,5),
            strides=(2,2),
            activations=(tf.nn.leaky_relu,tf.nn.leaky_relu)
        )
        D_optimizer=tf.keras.optimizers.Adam(1e-4)
       
        #checkpoints
        checkpoint = tf.train.Checkpoint(
            G_optimizer=G_optimizer, 
            D_optimizer=D_optimizer,
            G=G,
            D=D
            )

    #basic train step in one device
    with strategy.scope():
        def train_step(inputs):
            x,y=inputs
            noise=tf.random.normal(shape=(BATCH_SIZE_PER_REPLICA,NOISE_DIM))
            with tf.GradientTape() as g_tape,tf.GradientTape() as d_tape:
                #noise to images
                fake_images=G(noise,training=True)
                #juege by discriminator
                real_logits=D(x,training=True)
                fake_logits=D(fake_images,training=True)
                #compute loss
                gen_loss=g_loss(fake_logits)
                disc_loss=d_loss(real_logits,fake_logits)

            #compute gradients
            g_gradients=g_tape.gradient(gen_loss,G.trainable_variables)
            d_gradients=d_tape.gradient(disc_loss,D.trainable_variables)

            #apply gradients
            G_optimizer.apply_gradients(zip(g_gradients,G.trainable_variables))
            D_optimizer.apply_gradients(zip(d_gradients,D.trainable_variables))

            return gen_loss,disc_loss

    
    #distribute train_step use basic train step
    with strategy.scope():
        def dist_train_step(dataset_inputs):
            replica_losses=strategy.experimental_run_v2(fn=train_step,args=(dataset_inputs,))
            print("replica_losses:\n",replica_losses)
            # return strategy.reduce(reduce_op=tf.distribute.ReduceOp.SUM,value=replica_losses,axis=None)

        for epoch in range(EPOCHS):
            print("-----------EPOCH:",epoch)
            #epoch_loss=0.0
            num_batchs=0
            for records in train_dist_dataset:
                #epoch_loss+=dist_train_step(records)
                dist_train_step(records)
                num_batchs+=1
            # epoch_loss=epoch_loss/num_batchs

            # print("epoch_loss:",epoch_loss.numpy())
            # print("epoch_accuracy:",train_accuracy.result().numpy())

            #reset states
            # train_accuracy.reset_states()

            #save model
            checkpoint.save(CHECK_POINTS_PATH)



if __name__=="__main__":
    train()



