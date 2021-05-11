import tensorflow.compat.v1 as tf
import tflearn
import Build_Models.auto_encoder as AE
from Build_Models.auto_encoder import create_auto_encoder
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

batch_size=256

def main():

    X=np.load("inputs.npy")

    print(X.shape)
    # print(X[0],len(X),len(X[0]),len(X[0][0]))
    X=X.reshape([-1,8, 1, 23])
    # print(X[0],len(X),len(X[0]),len(X[0][0]))

    X, X_test, Y, Y_test = train_test_split(X, X, test_size=0.33, random_state=42)
    # print(X.shape)
    # print(X_test.shape)
    # print(Y.shape)
    # print(Y_test.shape)
    # print(len(X),len(X[0]),len(X[0][0]),X.shape)
    # print(len(X_test),len(X_test[0]),len(X_test[0][0]),X_test.shape)
    # print(len(Y),len(Y[0]),len(Y[0][0]),Y.shape)
    # print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]),Y_test.shape)

    model =AE.create_auto_encoder()

    model.fit({'input': X}, {'target': Y}, n_epoch=36,batch_size=batch_size,
    validation_set=({'input': X_test}, {'target': Y_test}),
    snapshot_step=1000,show_metric=True, run_id='convnet_mnist')



    # model.save("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")

    # df=pd.DataFrame(X[0].reshape(8, 23))
    # df.to_csv("./TrainingOutputs/inputs.csv")

    # encode_decode = model.predict(X[0].reshape(1,8, 1, 23))
    # output=np.array(encode_decode)
    # output[:] = output[:]>0.5
    # df=pd.DataFrame(output.reshape(8, 23))
    # df.to_csv("./TrainingOutputs/outputs.csv")

main()