import tensorflow.compat.v1 as tf

import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d
from tflearn.activations import relu , sigmoid
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from tflearn.layers.estimator import regression
import pandas as pd 
import numpy as np
batch_size=256

#######################################################################################################
def create_auto_encoder():
    # Building the network
    channel_size = [23, 32, 64, 64, 256, 256,256, 256,64, 64, 32,23]
    # betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
    betas =  [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(len(channel_size))]
    # print(betas[0])

    encoder_channel_size= [23, 32, 64, 64, 256, 256]
    AE = input_data(shape=[None,8, 1, 23], name='input')
    for i in range(len(encoder_channel_size)):
        AE = conv_2d(AE,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}")
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}")#,trainable=False
        AE = sigmoid(AE)
        # AE = AE + betas[i]
        # AE = relu(AE)

    decoder_channel_size= [256, 256,64, 64, 32,23]
    for i in range(len(decoder_channel_size)):
        AE = conv_2d(AE,decoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convDecoder_{i}",restore=False)
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeDecoder_{i}",restore=False)#,trainable=False
        AE = sigmoid(AE)
        # AE = AE + betas[i]
        # AE = relu(AE)

    AE = regression(AE, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target',restore=False)


    # creating the model
    auto_encoder = tflearn.DNN(AE,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/autoencoder/AE',
    checkpoint_path = './TrainingOutputs/autoencoder/AE/checkpoint')

    return auto_encoder