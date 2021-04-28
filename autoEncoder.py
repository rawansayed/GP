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
import pandas as pd 
import numpy as np
batch_size=256


# file_path = './paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# # file_path = './examples/eg_cls_on_target.episgt'
# input_data = Episgt(file_path, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X = np.expand_dims(x, axis=2)
# np.save("inputs.npy",X)
# np.save("labels.npy",y)

#######################################################################################################
def create_auto_encoder():
    # Building the network
    channel_size = [23, 32, 64, 64, 256, 256,256, 256,64, 64, 32,23]
    # betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
    betas =  [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(len(channel_size))]
    # print(betas[0])

    AE = input_data(shape=[None,8, 1, 23], name='input')
    for i in range(len(channel_size)):
        AE = conv_2d(AE,channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}")
        AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)#,trainable=False
        # AE = AE + betas[i]
        # AE = relu(AE)
        AE=sigmoid(AE)


    AE = regression(AE, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')


    # Training the network
    auto_encoder = tflearn.DNN(AE,tensorboard_verbose=0,
    tensorboard_dir = './TrainingAutoEncoder/autoencoder/AE',
    checkpoint_path = './TrainingAutoEncoder/autoencoder/AE/checkpoint')

    return auto_encoder

def main():

    X=np.load("inputs.npy")

    print(X.shape)
    # print(X[0],len(X),len(X[0]),len(X[0][0]))
    X=X.reshape([-1,8, 1, 23])
    # print(X[0],len(X),len(X[0]),len(X[0][0]))

    X, X_test, Y, Y_test = train_test_split(X, X, test_size=0.33, random_state=42)
    print(X.shape)
    print(X_test.shape)
    print(Y.shape)
    print(Y_test.shape)
    # print(len(X),len(X[0]),len(X[0][0]),X.shape)
    # print(len(X_test),len(X_test[0]),len(X_test[0][0]),X_test.shape)
    # print(len(Y),len(Y[0]),len(Y[0][0]),Y.shape)
    # print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]),Y_test.shape)

    model =create_auto_encoder()

    model.fit({'input': X}, {'target': Y}, n_epoch=36,batch_size=batch_size,
    validation_set=({'input': X_test}, {'target': Y_test}),
    snapshot_step=1000,show_metric=True, run_id='convnet_mnist')



    model.save("./TrainingAutoEncoder/autoencoder/model.tfl")

    df=pd.DataFrame(X[0].reshape(8, 23))
    df.to_csv("./TrainingAutoEncoder/inputs.csv")

    encode_decode = model.predict(X[0].reshape(1,8, 1, 23))
    output=np.array(encode_decode)
    output[:] = output[:]>0.5
    df=pd.DataFrame(output.reshape(8, 23))
    df.to_csv("./TrainingAutoEncoder/outputs.csv")

# main()