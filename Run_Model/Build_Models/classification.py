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





def attention(inputs, attention_size,restore=True, time_major=False, return_alphas=False,name='',trainable=True):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        restore:
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # if isinstance(inputs, tuple):
    #     # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    #     inputs = tf.concat(inputs, 2)

    # if time_major:
    #     # (T,B,D) => (B,T,D)
    #     inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    # hidden_size = inputs.shape[3].value  # D value - hidden size of the RNN layer

    # initializer = tf.random_normal_initializer(stddev=0.1)

    # # Trainable parameters
    # w_omega =vs.variable(f"{name}_w_omega", shape=[hidden_size, attention_size],
    #                 initializer=initializer,
    #                 trainable=trainable, restore=restore)
    # b_omega = vs.variable(f"{name}_b_omega", shape=[attention_size],
    #                 initializer=initializer,
    #                 trainable=trainable, restore=restore)
    # u_omega = vs.variable(f"{name}_u_omega", shape=[attention_size],
    #                 initializer=initializer,
    #                 trainable=trainable, restore=restore)
    


    # # w_omega = tf.get_variable(name=f"{name}_w_omega", shape=[hidden_size, attention_size], initializer=initializer)
    # # b_omega = tf.get_variable(name=f"{name}_b_omega", shape=[attention_size], initializer=initializer)
    # # u_omega = tf.get_variable(name=f"{name}_u_omega", shape=[attention_size], initializer=initializer)

    # with tf.name_scope(f"{name}_v"):
    #     # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #     #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    #     v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu = tf.tensordot(v, u_omega, axes=1, name=f"{name}_vu")  # (B,T) shape
    # alphas = tf.nn.softmax(vu, name=f"{name}_alphas")         # (B,T) shape

    # # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # output = tf.reduce_sum(  inputs * tf.expand_dims(alphas, -1),[])


    # if not return_alphas:
    #     return output
    # else:
    #     return output, alphas



def attention_CNN(inputs, attention_size,restore=True, time_major=False, return_alphas=False,name='',trainable=True):
    outputs=[]
    return outputs








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

        # x,y,z,a=CLS.shape
        # print(x,y,z,a)
        # CLS=reshape(CLS,[-1,z,a])
        # print(CLS.shape)
        # CLS=lstm(CLS,z*encoder_channel_size[i],activation='linear',restore=False,name=f'encoder_lstm_{i}')
        # print(CLS.shape)
        # CLS=reshape(CLS,[-1,y,z,a])

        # CLS = attention(CLS,encoder_channel_size[i],name = f"encoder_attention_{i}",trainable=False)
        
        # end each layer with relu activation layer
        CLS = activation(CLS,activation='relu', name=f'encoder_relu_{i}')
        # CLS = dropout(CLS,0.7,name=f'Dropout_{i}')



    # CLS = conv_2d(CLS,256*2, [1, 3],name=f"Temp_0",restore=False)
    # CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeTemp_0",restore=False)
    # CLS = activation(CLS,activation='relu', name=f'Temp_relu_0')

    # CLS = conv_2d(CLS,256*4, [1, 3],name=f"Temp_1",restore=False)
    # CLS = batch_normalization(CLS,decay=0,name=f"BatchNormalizeTemp_1",restore=False)
    # CLS = activation(CLS,activation='relu', name=f'Temp_relu_1')


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
        CLS = batch_normalization(CLS,decay=0.99,name=f"BatchNormalizeCls_{i}",trainable=False)

        # x,y,z,a=CLS.shape
        # print(x,y,z,a)
        # CLS=reshape(CLS,[-1,z,a])
        # print(CLS.shape)
        # CLS=lstm(CLS,z*cls_channel_size[i],activation='linear',restore=False,name=f'CLS_lstm_{i}')
        # print(CLS.shape)
        # CLS=reshape(CLS,[-1,y,z,a])

        # CLS = attention(CLS,encoder_channel_size[i],name=f"cls_attention_{i}",restore = False)

        #end each layer with relu activation layer
        CLS = activation(CLS,activation='relu', name=f'cls_relu_{i}')
        # CLS = dropout(CLS,0.8,name=f'Dropout_{i}')

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
        , loss='binary_crossentropy', name='target')
    # binary_crossentropy
    # categorical_crossentropy
    # roc_auc_score


    # creating the model
    model = tflearn.DNN(CLS,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/classification/CP',
    checkpoint_path = './TrainingOutputs/classification/CP/checkpoint')

    return model