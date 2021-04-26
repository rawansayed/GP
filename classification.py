import tensorflow.compat.v1 as tf
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


# file_path = './paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# # file_path = './examples/eg_cls_on_target.episgt'
# input_data = Episgt(file_path, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X = np.expand_dims(x, axis=2)
# np.save("file.npy",X)
# np.save()
#######################################################################################################
X=np.load("inputs.npy")
y=np.load("labels.npy")
print(X.shape)
print(y.shape)
# print(X[0],len(X),len(X[0]),len(X[0][0]))
X=X.reshape([-1,8, 1, 23])
y=y.reshape([-1,1, 1, 1])

# print(X[0],len(X),len(X[0]),len(X[0][0]))


X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X.shape)
print(X_test.shape)
Y=Y.reshape([-1,])
print(Y.shape)
Y_test=Y_test.reshape([-1,]) 
print(Y_test.shape)
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
    AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}")
    AE=sigmoid(AE)
    # AE = AE + betas[i]
    # AE = relu(AE)

cls_channel_size = [512, 512, 1024, 2]
for i in range(len(cls_channel_size)-1):
    if i==0:
        AE = conv_2d(AE,cls_channel_size[i], [1, 3], strides=2,bias=False,activation=None,name=f"convCls_{i}")
    if i==1:
        AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}")
    if i==2:
        AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}")#,padding='VALID'
    AE = batch_normalization(AE,decay=0.99,name=f"BatchNormalizeCls_{i}")
    # AE = AE + betas[i]
    # AE = relu(AE)
    AE=sigmoid(AE)

AE = conv_2d(AE,cls_channel_size[3], [1, 1],bias=False,activation=None,name="convCls_3")
# AE = AE + betas[i]
AE = tf.nn.softmax(AE)
print(AE.shape)
AE = tf.squeeze(AE, axis=[1, 2])[:, 1]

AE = regression(AE, optimizer='adam', learning_rate=0.0001, loss='sparse_categorical_crossentropy', name='target')


# Training the network
model = tflearn.DNN(AE,tensorboard_verbose=0,
tensorboard_dir = 'MNIST_tflearn_board/',
checkpoint_path = 'MNIST_tflearn_checkpoints/checkpoint')


model.load("./TrainingAutoEncoder/autoencoder/model.tfl")




model.fit({'input': X}, {'target': Y}, n_epoch=120,batch_size=batch_size,
validation_set=({'input': X_test}, {'target': Y_test}),
snapshot_step=1000,show_metric=True, run_id='convnet_mnist')

model.save("./TrainingAutoEncoder/classification/ClassificationModel.tfl")