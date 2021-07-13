from tflearn.metrics import Accuracy
import tensorflow.compat.v1 as tf
import tflearn
from tflearn.layers.core import input_data , activation
from tflearn.layers.conv import conv_2d, max_pool_1d
from tflearn.activations import relu , sigmoid, softmax
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from tflearn.metrics import accuracy
from tflearn.layers.core import dropout
import pandas as pd 
import numpy as np

batch_size=256

def create_regression_model():
   # Building the network
    channel_size=[8, 32, 64, 64, 256, 256,512, 512, 1024, 1]

    
    encoder_channel_size= [32, 64, 64, 256, 256]
    betas =  [
        tf.Variable(0.0 * tf.ones(encoder_channel_size[i]), name=f'beta_{i}') for i in range(len(encoder_channel_size))
    ] 

    # layer that take the input
    REG = input_data(shape=[None,1, 23, 8], name='input')

    # Start the encoder layers 
    # all these layer have the the trainable pararmetar = False so it doesnot overwrite our trained weights
    for i in range(len(encoder_channel_size)):
        # creating the convolation layers
        if i == 1 or i == 3:
            REG = conv_2d(REG,encoder_channel_size[i], [1, 3],strides=2,name=f"convEncoder_{i}",trainable=False)
        else:
            REG = conv_2d(REG,encoder_channel_size[i], [1, 3],name=f"convEncoder_{i}",trainable=False)
        # creating the batch normalization layers
        REG = batch_normalization(REG,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)     
        REG = REG + betas[i]
        # end each layer with relu activation layer
        REG = activation(REG,activation='relu', name=f'encoder_relu_{i}')

    # Regression Layers
    # All these layers have parameter restore = False so it doesn't restor these layer from the encoder files to do not make errors
    # of course when we will run these files after trainig we will need to change this parameter
    reg_channel_size = [512, 512, 1024,1]
    for i in range(len(reg_channel_size)-1):
        # creating the convolation layers 
        if i==0:
            REG = conv_2d(REG,reg_channel_size[i], [1, 3], strides=2,name=f"convReg_{i}",restore=False)
        if i==1:
            REG = conv_2d(REG,reg_channel_size[i], [1, 3],name=f"convReg_{i}",restore=False)
        if i==2:
            REG = conv_2d(REG,reg_channel_size[i], [1, 3],name=f"convReg_{i}",restore=False,padding='VALID')
        # creating the batchnormalization layers 
        REG = batch_normalization(REG,decay=0.99,name=f"BatchNormalizeReg_{i}",restore=False)
        #end each layer with relu activation layer
        REG = activation(REG,activation='relu', name=f'reg_relu_{i}')

    # for the last layer we will only do convolution then sigmoid activation 
    REG = conv_2d(REG,reg_channel_size[3], [1, 1],name="convReg_3",restore=False)
    
    # and end it with squeeze function so the output end in a shape (-1,)
    
    REG = tf.squeeze(REG, axis=[1, 2, 3])

    # we define our optimizer and loss functions and learning rate in the regression layer 
    REG = regression(REG, optimizer='adam', learning_rate=0.001
        , loss='mean_square', name='target', restore=False)


    # creating the model
    model = tflearn.DNN(REG,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/Regression/CP',
    checkpoint_path = './TrainingOutputs/Regression/CP/checkpoint')

    return model