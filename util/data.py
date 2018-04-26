import pandas as pd
import numpy as np
import sklearn.preprocessing as skprep
import matplotlib.pyplot as plt



def load_mnist(reshape=False):
    # generate data
    train_frame = pd.read_csv("../data/MNIST/train.csv")
    test_frame = pd.read_csv("../data/MNIST/test.csv")

    # pop the labels and one-hot coding
    train_labels_frame = train_frame.pop("label")

    # get values
    # one-hot on labels
    X_train = train_frame.astype(np.float32).values
    y_train = pd.get_dummies(data=train_labels_frame).values
    X_test = test_frame.astype(np.float32).values

    if reshape==False:
        return X_train,y_train,X_test
    else:
        # trans the shape to (batch,time_steps,input_size)
        X_train = np.reshape(X_train, newshape=(-1, 28, 28))
        X_test = np.reshape(X_test, newshape=(-1, 28, 28))
        return X_train,y_train,X_test

def standard(X_train,X_test):
    X_train=X_train/256
    X_test=X_test/256
    return X_train,X_test


if __name__=="__main__":
    X_train, y_train, X_test=load_mnist(reshape=False)
    print(X_train[0])
    X_train,X_test=standard(X_train=X_train,X_test=X_test)
    print(X_train[0])
    print(y_train[0])
    plt.imshow(np.reshape(X_train[0],[28,28]))
    plt.show()