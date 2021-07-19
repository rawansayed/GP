import tflearn
from tflearn.layers.core import input_data, activation
from tflearn.layers.conv import conv_2d , conv_2d_transpose
from tflearn.activations import relu , sigmoid
from tflearn.metrics import binary_accuracy_op ,WeightedR2,R2,accuracy
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from tflearn.layers.estimator import regression
import pandas as pd 
import numpy as np
import tensorflow.compat.v1 as tf
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
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}")#,trainable=False

        AE = AE + betas[i]
        # AE = tf.math.add( AE, betas, name=f'add_beta_{i}')

        AE = attention(AE,encoder_channel_size[i],name = f"encoder_attention_{i}")

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
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeDecoder_{i}")#,trainable=False
        
        AE = attention(AE,decoder_channel_size[i],name = f"decoder_attention_{i}")
        
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
    AE = regression(AE, optimizer='adam',metric=accuracy()
    , learning_rate=0.001, loss='binary_crossentropy',name='target')
    # weak_cross_entropy_2d 60% 64 64
    # binary_crossentropy 60% 64 64
    # categorical_crossentropy (LOL)


    # creating the model
    auto_encoder = tflearn.DNN(AE,tensorboard_verbose=0,
    tensorboard_dir = './TrainingOutputs/autoencoder/CP',
    checkpoint_path = './TrainingOutputs/autoencoder/CP/checkpoint')

    return auto_encoder