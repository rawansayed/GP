import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.disable_v2_behavior()
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d
from tflearn.activations import relu , sigmoid, softmax
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from sklearn.model_selection import train_test_split
from tflearn.layers.estimator import regression
import pandas as pd 
import numpy as np
batch_size=256

from Build_Models.classification import create_classification_model


def main():
    tf.reset_default_graph()
    X=np.load("inputs.npy")
    y=np.load("labels.npy")
    # print(X.shape)
    # print(y.shape)
    # print(X[0],len(X),len(X[0]),len(X[0][0]))
    X=X.reshape([-1,8, 1, 23])
    y=y.reshape([-1,1, 1, 1])

    # print(X[0],len(X),len(X[0]),len(X[0][0]))
    # for i in range(36*6):
    #     print(array_of_weights_in_all_layers[i])
    X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # print(X.shape)
    # print(X_test.shape)
    Y=Y.reshape([-1,])
    # print(Y.shape)
    Y_test=Y_test.reshape([-1,]) 
    # print(Y_test.shape)
    # print(len(X),len(X[0]),len(X[0][0]),X.shape)
    # print(len(X_test),len(X_test[0]),len(X_test[0][0]),X_test.shape)
    # print(len(Y),len(Y[0]),len(Y[0][0]),Y.shape)
    # print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]),Y_test.shape)

    model=create_classification_model()




    model.fit({'input': X}, {'target': Y}, n_epoch=25,batch_size=batch_size,
    validation_set=({'input': X_test}, {'target': Y_test}),
    snapshot_step=1000,show_metric=True, run_id='convnet_mnist')

    # model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")

main()

