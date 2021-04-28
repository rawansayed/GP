import tensorflow.compat.v1 as tf
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d
from tflearn.activations import relu , sigmoid
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from sklearn.model_selection import train_test_split
from tflearn.layers.estimator import regression
import tflearn
import pandas as pd
import numpy as np
X=np.load('./TrainingOutputs/eg_1_cls_on_target.episgt.npy', allow_pickle=True)
# print(X[0],len(X),len(X[0]),len(X[0][0]))
X=X.reshape([-1,1, 23, 8])
# print(X[0],len(X),len(X[0]),len(X[0][0]))

X, X_test, Y, Y_test = train_test_split(X, X, test_size=0.33, random_state=42)
print(len(X),len(X[0]),len(X[0][0]))
print(len(X_test),len(X_test[0]),len(X_test[0][0]))
print(len(Y),len(Y[0]),len(Y[0][0]))
print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]))
# Building the network

channel_size = [8, 32, 64, 64, 256, 256,256, 256,64, 64, 32,8]
betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
print(betas[0])

AE = input_data(shape=[None,1, 23, 8], name='input')
for i in range(len(channel_size)):
    AE = conv_2d(AE,channel_size[i], [1, 3],bias=False,activation=None)
    AE = batch_normalization(AE,decay=0)#,beta=betas[i])
    AE = sigmoid(AE)

AE = regression(AE, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')


# Training the network
model = tflearn.DNN(AE,tensorboard_verbose=0,
tensorboard_dir = 'MNIST_tflearn_board/',
checkpoint_path = 'MNIST_tflearn_checkpoints/checkpoint')

model=model.load("./TrainingOutputs/model.tfl")


encode_decode = model.predict(X[0].reshape(1,1,23,8))
output=np.array(encode_decode)
output[:] = output[:]>0.5
df=pd.DataFrame(output)
df=pd.concat(df,X[0])

df.to_csv("./TrainingOutputs/filename.csv")