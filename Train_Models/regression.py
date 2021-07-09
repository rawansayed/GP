from Build_Models.regression import create_regression_model
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
import tflearn
from sklearn.metrics import roc_auc_score

batch_size=256

def main():
    # loading data
    X=np.load("D:\BIOMEDICAL ENGINEERING\GP\GP\inputs_reg.npy")
    y=np.load("D:\BIOMEDICAL ENGINEERING\GP\GP\labels_reg.npy")
    X = X.transpose([0, 2, 3, 1])

    # Creating train and development and test data
    X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.1, random_state=42)

    
    # create model from classification
    model = create_regression_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/Regression_Model/model.tfl")



    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=20,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True)
    
    # save the model
    model.save("./TrainingOutputs/Regression/regModel/RegressionModel.tfl")

    
main()