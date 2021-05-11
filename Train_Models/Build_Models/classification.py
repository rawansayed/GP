import tensorflow.compat.v1 as tf

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

def create_classification_model():
    # Building the network
    channel_size=[23, 32, 64, 64, 256, 256,512, 512, 1024, 2]
    # betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
    betas =  [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(len(channel_size))]
    # print(betas[0])
    AE = input_data(shape=[None,8, 1, 23], name='input')

    encoder_channel_size = [23, 32, 64, 64, 256, 256]
    for i in range(len(encoder_channel_size)):
        # array_of_weights[i] = tf.get_variable('get_variable'
        # , dtype=tf.float32, initializer=array_of_weights[i])
        # print(array_of_weights[i].value())
        # AE,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}"
        AE = conv_2d(AE,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}")#,weights_init=array_of_weights[i])
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)
        AE=sigmoid(AE)
        # AE = AE + betas[i]
        # AE = relu(AE)



    cls_channel_size = [512, 512, 1024,2]
    for i in range(len(cls_channel_size)-1):
        if i==0:
            
            AE = conv_2d(AE,cls_channel_size[i], [1, 3], strides=2,bias=False,activation=None,name=f"convCls_{i}",restore=False)
        if i==1:
            
            AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)
        if i==2:
            
            AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)#,padding='VALID'
        AE = batch_normalization(AE,decay=0.99,name=f"BatchNormalizeCls_{i}",restore=False)
        # AE = AE + betas[i]
        # AE = relu(AE)
        AE=sigmoid(AE)

    # AE=AE.reshape([None,1,1,2])
    AE = conv_2d(AE,cls_channel_size[3], [1, 1],bias=False,activation=None,name="convCls_3",restore=False)
    # AE = AE + betas[i]
    AE = tf.nn.softmax(AE)

    # AE.set_shape([None,1,1,2])
    print(AE.shape)
    AE=tf.reshape(AE, (-1,1,1,8))
    print(AE.shape)

    AE = tf.squeeze(AE, axis=[1, 2])[:, 1]

    # print(AE.shape)
    # AE = tf.squeeze(AE, axis=[1, 2])[:, 1]

    AE = regression(AE, optimizer='adam', learning_rate=0.0001
        , loss='categorical_crossentropy', name='target', restore=False)


    # creating the model
    model = tflearn.DNN(AE,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/classification/AE',
    checkpoint_path = './TrainingOutputs/classification/AE/checkpoint')

    return model