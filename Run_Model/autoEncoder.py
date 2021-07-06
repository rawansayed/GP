from sklearn.model_selection import train_test_split
from Build_Models.auto_encoder import create_auto_encoder
import numpy as np
import pandas as pd

def accuracyforAutoEncoder(y_pred,Y_output):

    return

batch_size=256
def main():
    # loading data
    X=np.load("DATA.npy")
    # X=X.reshape([-1,8, 1, 23])
    X = X.transpose([0, 2, 3, 1])
    # print(X[0])
    # print(X.shape)

    # Creating train and development data  // I did not make test data
    X, X_dev, Y, Y_dev = train_test_split(X, X, test_size=0.33, random_state=42)
    # print(X.shape)
    # print(X_dev.shape)
    # print(Y.shape)
    # print(Y_dev.shape)
    X = X 
    Y=X
    X_dev, X_test= train_test_split(X_dev, test_size=0.02, random_state=42)
    
    X_dev = X_dev 
    Y_dev = X_dev

    Y_test=X_dev
    
    # Creating model
    model =create_auto_encoder()
    
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")

    # print(accuracy())
    # print(accuracy())
    # print(accuracy())

main()