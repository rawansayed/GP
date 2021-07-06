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
import pandas as pd 
import numpy as np
batch_size=256

def create_classification_model():
    # Building the network
    channel_size=[8, 32, 64, 64, 256, 256,512, 512, 1024, 2]

    
    encoder_channel_size= [32, 64, 64, 256, 256]
    betas =  [
        tf.Variable(0.0 * tf.ones(encoder_channel_size[i]), name=f'beta_{i}') for i in range(len(encoder_channel_size))
    ]
    # layer that take the input
    CLS = input_data(shape=[None,1, 23, 8], name='input')
    # Start the encoder layers 
    # all these layer have the the trainable pararmetar =False so it doesnot overwrite our trained weights
    for i in range(len(encoder_channel_size)):
        # creating the convolation layers
        if i == 1 or i == 3:
            CLS = conv_2d(CLS,encoder_channel_size[i], [1, 3],strides=2,name=f"convEncoder_{i}",trainable=False)
        else:
            CLS = conv_2d(CLS,encoder_channel_size[i], [1, 3],name=f"convEncoder_{i}",trainable=False)
        # creating the batch normalization layers
        CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)
        
        CLS = CLS + betas[i]
        # end each layer with relu activation layer
        CLS = activation(CLS,activation='relu', name=f'encoder_relu_{i}')


    # Classifiction Layers
    # All these layers have parameter restore = False so it doesn't restor these layer from the encoder files to do not make errors
    # of course when we will run these files after trainig we will need to change this parameter
    cls_channel_size = [512, 512, 1024,2]
    for i in range(len(cls_channel_size)-1):
        # creating the convolation layers 
        if i==0:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3], strides=2,name=f"convCls_{i}")
        if i==1:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3],name=f"convCls_{i}")
        if i==2:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3],name=f"convCls_{i}",padding='VALID')
        # creating the batchnormalization layers 
        CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_{i}")
        #end each layer with relu activation layer
        CLS = activation(CLS,activation='relu', name=f'cls_relu_{i}')

    # for the last layer we will only do convolution then sigmoid activation 
    CLS = conv_2d(CLS,cls_channel_size[3], [1, 1],name="convCls_3")

    # #########################################################
    # CLS = conv_2d(CLS,1, [1, 1],name="convCls_3",restore=False)
    # CLS = activation(CLS,activation='leakyrelu', name=f'cls_sigmoid_3')
    
    # CLS = conv_2d(CLS,1, [1, 1],name="convCls_4",restore=False)
    # CLS = activation(CLS,activation='sigmoid', name=f'cls_sigmoid_4')
    # CLS = tf.reshape(CLS,[-1,])
    # #########################################################

    CLS = activation(CLS,activation='softmax', name=f'cls_sigmoid_3')
    # # # #and end it with squeeze function so the output end in a shape (-1,)
    CLS = tf.squeeze(CLS, axis=[1, 2])[:, 1]

    def loss(y_pred,y_true):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels= y_true))
        # return tf.nn.sigmoid_cross_entropy_with_logits(labels= y_true, logits= y_pred)

    # we define our optimizer and loss functions and learning rate in the regression layer 
    CLS = regression(CLS, optimizer='adam', learning_rate=0.001,metric=accuracy()
        , loss='binary_crossentropy', name='target', )
    # binary_crossentropy
    # categorical_crossentropy
    # roc_auc_score


    # creating the model
    model = tflearn.DNN(CLS,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/classification/CP',
    checkpoint_path = './TrainingOutputs/classification/CP/checkpoint')

    return model