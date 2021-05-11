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
import pandas as pd 
import numpy as np
batch_size=256

def create_classification_model():
    # Building the network
    channel_size=[8, 32, 64, 64, 256, 256,512, 512, 1024, 2]


    encoder_channel_size= [8, 32, 64, 64, 256, 256]
    CLS = input_data(shape=[None,1, 23, 8], name='input')
    for i in range(len(encoder_channel_size)):
        if i == 1 or i == 3:
            CLS = conv_2d(CLS,encoder_channel_size[i], [1, 3],strides=2,bias=False,activation=None,name=f"convEncoder_{i}")
        else:
            CLS = conv_2d(CLS,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}")
        CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)
        CLS = activation(CLS,activation='relu', name=f'encoder_sigmoid_{i}')


    cls_channel_size = [512, 512, 1024,2]
    for i in range(len(cls_channel_size)-1):
        if i==0:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3], strides=2,bias=False,activation=None,name=f"convCls_{i}",restore=False)
        if i==1:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)
        if i==2:
            CLS = conv_2d(CLS,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False,padding='VALID')
        CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_{i}",restore=False)
        CLS = activation(CLS,activation='relu', name=f'cls_relu_{i}')

    CLS = conv_2d(CLS,cls_channel_size[3], [1, 1],bias=False,activation=None,name="convCls_3",restore=False)
    CLS = activation(CLS,activation='softmax', name=f'cls_sigmoid_3')
    CLS = tf.squeeze(CLS, axis=[1, 2])[:, 1]


    CLS = regression(CLS, optimizer='adam', learning_rate=0.0001
        , loss='binary_crossentropy', name='target', restore=False)


    # creating the model
    model = tflearn.DNN(CLS,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/classification/CP',
    checkpoint_path = './TrainingOutputs/classification/CP/checkpoint')

    return model