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

from tflearn import variables as vs


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

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[3].value  # D value - hidden size of the RNN layer

    initializer = tf.random_normal_initializer(stddev=0.1)

    # Trainable parameters
    w_omega =vs.variable(f"{name}_w_omega", shape=[hidden_size, attention_size],
                    initializer=initializer,
                    trainable=trainable, restore=restore)
    b_omega = vs.variable(f"{name}_b_omega", shape=[attention_size],
                    initializer=initializer,
                    trainable=trainable, restore=restore)
    u_omega = vs.variable(f"{name}_u_omega", shape=[attention_size],
                    initializer=initializer,
                    trainable=trainable, restore=restore)

    with tf.name_scope(f"{name}_v"):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name=f"{name}_vu")  # (B,T) shape
    alphas = tf.nn.softmax(vu, name=f"{name}_alphas")         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(  inputs * tf.expand_dims(alphas, -1),[])


    if not return_alphas:
        return output
    else:
        return output, alphas

def create_regression_model():
   # Building the network
    channel_size = [8, 32, 64, 64, 256, 256,512, 512, 1024, 1]

    
    encoder_channel_size = [32, 64, 64, 256, 256]
    betas =  [
        tf.Variable(0.0 * tf.ones(encoder_channel_size[i]), name=f'beta_{i}') for i in range(len(encoder_channel_size))
    ] 

    # layer that take the input
    REG = input_data(shape = [None, 1, 23, 8], name = 'input')

    # Start the encoder layers 
    # all these layer have the the trainable pararmetar = False so it doesnot overwrite our trained weights
    for i in range(len(encoder_channel_size)):
        # creating the convolation layers
        if i == 1 or i == 3:
            REG = conv_2d(REG,encoder_channel_size[i], [1, 3],strides=2,name=f"convEncoder_{i}")
        else:
            REG = conv_2d(REG,encoder_channel_size[i], [1, 3],name=f"convEncoder_{i}")
        # creating the batch normalization layers
        REG = batch_normalization(REG, decay = 0, name = f"BatchNormalizeEncoder_{i}", trainable=False)     
        REG = REG + betas[i]

        REG = attention(REG,encoder_channel_size[i],name = f"encoder_attention_{i}")

        # end each layer with relu activation layer
        REG = activation(REG, activation = 'relu', name = f'encoder_relu_{i}')

    # Regression Layers
    # All these layers have parameter restore = False so it doesn't restor these layer from the encoder files to do not make errors
    # of course when we will run these files after trainig we will need to change this parameter
    reg_channel_size = [512, 512, 1024,1]
    for i in range(len(reg_channel_size)-1):
        # creating the convolation layers 
        if i==0:
            REG = conv_2d(REG, reg_channel_size[i], [1, 3], strides=2, name = f"convReg_{i}", restore = False)
        if i==1:
            REG = conv_2d(REG, reg_channel_size[i], [1, 3], name = f"convReg_{i}", restore = False)
        if i==2:
            REG = conv_2d(REG, reg_channel_size[i], [1, 3], name = f"convReg_{i}", restore = False, padding = 'VALID')
        # creating the batchnormalization layers 
        REG = batch_normalization(REG, decay = 0.99, name = f"BatchNormalizeReg_{i}", restore = False,trainable=False)
        
        REG = attention(REG,encoder_channel_size[i],name=f"reg_attention_{i}",restore = False)
        
        #end each layer with relu activation layer
        REG = activation(REG, activation = 'relu', name = f'reg_relu_{i}')
        
    
    # for the last layer we will only do convolution then sigmoid activation 
    REG = conv_2d(REG, reg_channel_size[3], [1, 1], name = "convReg_3", restore = False)
    
    # and end it with squeeze function so the output end in a shape (-1,)
    
    REG = tf.squeeze(REG, axis = [1, 2, 3])

    # we define our optimizer and loss functions and learning rate in the regression layer 
    REG = regression(REG, optimizer='adam', learning_rate = 0.001
        , loss = 'mean_square', name = 'target', restore = False)


    # creating the model
    model = tflearn.DNN(REG, tensorboard_verbose = 0,
    tensorboard_dir = './checkpoints/Regression/CP',
    checkpoint_path = './checkpoints/Regression/CP/checkpoint')

    return model