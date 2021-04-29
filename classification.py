import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

from autoEncoder import create_auto_encoder

def get_all_tensor_names(filename):
    l=[i.name for i in tf.get_default_graph().get_operations()]
    # l = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    a_file = open(filename, "w")
    np.savetxt(a_file, l, delimiter=',', fmt='%s')
    a_file.close()

#######################################################################################################

# tf.reset_default_graph()
# auto_encoder = create_auto_encoder()
# auto_encoder.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl",weights_only=True)

# tf222= tf.get_default_graph()


# get_all_tensor_names in side the model
# get_all_tensor_names("./testFiles/testEncoder.txt")

###### to get cerain tensor using name
# array_of_weights_conv =[]
# array_of_weights_BN =[]
# for i in range(len(encoder_channel_size)):
#     array_of_weights_conv.append( tf.get_default_graph().get_tensor_by_name(f"convEncoder_{i}/Conv2D:0").W)
#     array_of_weights_BN.append(   tf.get_default_graph().get_tensor_by_name(f"BatchNormalizeEncoder_{i}/cond/Identity_1:0").W)

##### To debug tensors
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl", tensor_name=None, all_tensors=True)




# ########################################################################################################################
tf.reset_default_graph()
X=np.load("inputs.npy")
y=np.load("labels.npy")
# print(X.shape)
# print(y.shape)
# print(X[0],len(X),len(X[0]),len(X[0][0]))
X=X.reshape([-1,8, 1, 23])
y=y.reshape([-1,1, 1, 1])

# print(X[0],len(X),len(X[0]),len(X[0][0]))


X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X.shape)
# print(X_test.shape)
Y=Y.reshape([-1,])
# print(Y.shape)
Y_test=Y_test.reshape([-1,]) 
# print(Y_test.shape)
# print(len(X),len(X[0]),len(X[0][0]),X.shape)
# print(len(X_test),len(X_test[0]),len(X_test[0][0]),X_test.shape)
# print(len(Y),len(Y[0]),len(Y[0][0]),Y.shape)
# print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]),Y_test.shape)




# Building the network
channel_size=[23, 32, 64, 64, 256, 256,512, 512, 1024, 2]
# betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
betas =  [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(len(channel_size))]
# print(betas[0])
AE = input_data(shape=[None,8, 1, 23], name='input')

encoder_channel_size = [23, 32, 64, 64, 256, 256]
for i in range(len(encoder_channel_size)):
    AE = conv_2d(AE,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}")
    AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)
    AE=sigmoid(AE)
    # AE = AE + betas[i]
    # AE = relu(AE)

cls_channel_size = [512, 512, 1024,2]
for i in range(len(cls_channel_size)-1):
    if i==0:
        continue
        # AE = conv_2d(AE,cls_channel_size[i], [1, 3], strides=2,bias=False,activation=None,name=f"convCls_{i}",restore=False)
    if i==1:
        continue
        # AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)
    if i==2:
        continue
        # AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)#,padding='VALID'
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


# Training the network
model = tflearn.DNN(AE,tensorboard_verbose=0,
tensorboard_dir = './TrainingOutputs/classification/AE',
checkpoint_path = './TrainingOutputs/classification/AE/checkpoint')

get_all_tensor_names("./testFiles/testCls.txt")

# weight saving and loading resources
# Keywords:
# dynamically assign weights to a layer
# initialize layer weights from a numpy array
# how to write a layer initializer function in tensorflow
# what is a feed_dict in tensorflow
# Process:
# extract weights from layer as numpy array
# Either initialize or dynamically assign weights to new layers

model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl",weights_only=True)

# encoder_channel_size = [23, 32, 64, 64, 256, 256]
# for i in range(len(encoder_channel_size)):
#     model.set_weights(tf.get_default_graph().get_tensor_by_name(f"convEncoder_{i}:0").W
#     , array_of_weights_conv[i])
#     model.set_weights(tf.get_default_graph().get_tensor_by_name(f"BatchNormalizeEncoder_{i}:0").W
#     , array_of_weights_BN[i])



model.fit({'input': X}, {'target': Y}, n_epoch=25,batch_size=batch_size,
validation_set=({'input': X_test}, {'target': Y_test}),
snapshot_step=1000,show_metric=True, run_id='convnet_mnist')

model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")


