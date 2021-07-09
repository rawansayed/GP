from Build_Models.regression import create_regression_model
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape
import tensorflow.compat.v1 as tf
import tflearn
from scipy.stats import spearmanr

batch_size=256

def main():
    # loading data
    x=np.load("inputs_reg.npy")
    y=np.load("labels_reg.npy")
    x = x.transpose([0, 2, 3, 1])

    # Creating train and development and test data
    X, X_test, Y, Y_test = train_test_split(x, y, test_size=0.33, random_state=42,shuffle=True)
    X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

    class MonitorCallback(tflearn.callbacks.Callback):
        def on_epoch_end(self, training_state):
            pred_prob=model.predict(X_test)
            # correlation1 = spearmanr(Y_test, pred_prob)
            # print(f'test Spearman Corr value={correlation1:.6f}')
            print("test Spearman Corr value :",spearmanr(Y_test, pred_prob))
           
            pred_prob=model.predict(x) 
            # correlation2 = spearmanr(y, pred_prob)
            # print(f'whole Spearman Corr value={correlation2:.6f}')
            print("whole Spearman Corr value :",spearmanr(y, pred_prob))
            
    
    # create model from classification
    model = create_regression_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/Regression_Model/model.tfl")

    monitorCallback = MonitorCallback()
    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=20,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True,callbacks=monitorCallback)
    
    # save the model
    model.save("./TrainingOutputs/Regression/regModel/RegressionModel.tfl")

main()