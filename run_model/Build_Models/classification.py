from token import SLASH
from typing import ClassVar
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
from tflearn.layers.core import dropout
from tflearn.metrics import accuracy
from tflearn.layers.core import reshape
import pandas as pd 
import numpy as np
from tflearn import variables as vs
from tflearn.layers.recurrent import lstm
# from layers import augmented_conv2d
batch_size=256


def create_classification_model():
    # Building the network
    channel_size=[8, 32, 64, 64, 256, 256,512, 512, 1024, 2]

    # Building the input layer
    CLS = input_data(shape=[None,1, 23, 8], name='input')

    

    encoder_channel_size= [32, 64, 64, 256, 256]
    # Start the encoder layers 
    
    # creating the convolation layers
    CLS = conv_2d(CLS,encoder_channel_size[0], [1, 3],name=f"convEncoder_0")
    # creating the batch normalization layers
    CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_0",trainable=False)
    # end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'encoder_relu_0')

    # creating the convolation layers
    CLS = conv_2d(CLS,encoder_channel_size[1], [1, 3],strides=2,name=f"convEncoder_1")
    # creating the batch normalization layers
    CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_1",trainable=False)
    # end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'encoder_relu_1')

    # creating the convolation layers
    CLS = conv_2d(CLS,encoder_channel_size[2], [1, 3],name=f"convEncoder_2")
    # creating the batch normalization layers
    CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_2",trainable=False)
    # end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'encoder_relu_2')

    # creating the convolation layers
    CLS = conv_2d(CLS,encoder_channel_size[3], [1, 3],strides=2,name=f"convEncoder_3")
    # creating the batch normalization layers
    CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_3",trainable=False)
    # end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'encoder_relu_3')

    # creating the convolation layers
    CLS = conv_2d(CLS,encoder_channel_size[4], [1, 3],name=f"convEncoder_4")
    # creating the batch normalization layers
    CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_4",trainable=False)
    # end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'encoder_relu_4')
        
    
    
        

    # Classifiction Layers
    cls_channel_size = [512, 512, 1024,2]

    # creating the convolation layers 
    CLS = conv_2d(CLS,cls_channel_size[0], [1, 3], strides=2,name=f"convCls_0")
    # creating the batchnormalization layers 
    CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_0",trainable=False)
    #end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'cls_relu_0')

    # creating the convolation layers 
    CLS = conv_2d(CLS,cls_channel_size[1], [1, 3],name=f"convCls_1")
    # creating the batchnormalization layers 
    CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_1",trainable=False)
    #end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'cls_relu_1')

    # creating the convolation layers 
    CLS = conv_2d(CLS,cls_channel_size[2], [1, 3],name=f"convCls_2",padding='VALID')
    # creating the batchnormalization layers 
    CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_2",trainable=False)
    #end each layer with relu activation layer
    CLS = activation(CLS,activation='relu', name=f'cls_relu_2')


    # for the last layer we will only do convolution then sigmoid activation 
    CLS = conv_2d(CLS,cls_channel_size[3], [1, 1],name="convCls_3")
    CLS = activation(CLS,activation='softmax', name=f'cls_sigmoid_3')

    #and end it with squeeze function so the output end in a shape (-1,)
    CLS = tf.squeeze(CLS, axis=[1, 2])[:, 1]
    

    # creating the model
    model = tflearn.DNN(CLS,tensorboard_verbose=0,
    tensorboard_dir = './checkpoints/classification/CP/S1',
    checkpoint_path = './checkpoints/classification/CP/S1/checkpoint')

    return model