import tflearn
from tflearn.layers.core import input_data, activation
from tflearn.layers.conv import conv_2d , conv_2d_transpose
from tflearn.activations import relu , sigmoid
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from tflearn.layers.estimator import regression
import pandas as pd 
import numpy as np
import tensorflow.compat.v1 as tf

batch_size=256

#######################################################################################################
def create_auto_encoder():
    # Building the network
    channel_size = [8, 32, 64, 64, 256, 256,256, 256,64, 64, 32,8]


    encoder_channel_size= [32, 64, 64, 256, 256]
    betas =  [
        tf.Variable(0.0 * tf.ones(encoder_channel_size[i]), name=f'beta_{i}') for i in range(len(encoder_channel_size))
    ]
    
    # layer that take the input
    AE = input_data(shape=[None,1, 23, 8], name='input')
    # Start the encoder layers 
    for i in range(len(encoder_channel_size)):
        # creating the convolation layers
        if i == 1 or i == 3:
            AE = conv_2d(AE,encoder_channel_size[i], [1, 3],strides=2,name=f"convEncoder_{i}")
        else:
            AE = conv_2d(AE,encoder_channel_size[i], [1, 3],name=f"convEncoder_{i}")
        # creating the batch normalization layers
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)

        AE = AE + betas[i]
        # AE = tf.math.add( AE, betas, name=f'add_beta_{i}')

        # end each layer with relu activation layer
        AE = activation(AE,activation='relu', name=f'encoder_relu_{i}')


    decoder_channel_size= [256, 256,64, 64, 32,8]
    # Start the encoder layers
    for i in range(len(decoder_channel_size)-1):
        # creating the convolation transpose layers to becom an anti encoder
        if i == 0:
            AE = conv_2d_transpose(AE,decoder_channel_size[i], [1, 3],[1,6],name=f"convTransDecoder_{i}")
        if i == 1:
            AE = conv_2d_transpose(AE,decoder_channel_size[i], [1, 3],[1,12],strides=2,name=f"convTransDecoder_{i}")
        if i == 2:
            AE = conv_2d_transpose(AE,decoder_channel_size[i], [1, 3],[1,12],name=f"convTransDecoder_{i}")
        if i == 3:
            AE = conv_2d_transpose(AE,decoder_channel_size[i], [1, 3],[1,23],strides=2,name=f"convTransDecoder_{i}")
        if i == 4:
            AE = conv_2d_transpose(AE,decoder_channel_size[i], [1, 3],[1,23],name=f"convTransDecoder_{i}")
        # creating the batch normalization layers
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeDecoder_{i}",trainable=False)
        # end each layer with sigmoid activation layer
        AE = activation(AE,activation='relu', name=f'decoder_sigmoid_{i}')
    
    # the last layer will be only made of conv_transpose and a relu activation layer
    AE = conv_2d_transpose(AE,decoder_channel_size[5], [1, 3],[1,23],name=f"convTransDecoder_5")
    AE = activation(AE,activation='relu', name=f'decoder_relu_5')

    #to definr loss
    # def cross_entropy_loss(y_pred, y_true):
    #     cross_entropy = tf.reduce_mean(
    #         -tf.reduce_sum(
    #             y_true * tf.log(tf.clip_by_value(y_pred,1e-10,1.0))
    #             , reduction_indices=[1])
    #             )
    #     return cross_entropy

    # we define our optimizer and loss functions and learning rate in the regression layer 
    AE = regression(AE, optimizer='adam', learning_rate=0.001, loss='weak_cross_entropy_2d',name='target')



    # creating the model
    auto_encoder = tflearn.DNN(AE,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/autoencoder/CP',
    checkpoint_path = './TrainingOutputs/autoencoder/CP/checkpoint')

    return auto_encoder