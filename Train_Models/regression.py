from Build_Models.regression import create_regression_model
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape
import tensorflow.compat.v1 as tf
import tflearn
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

batch_size=256

def main():
    # loading data
    x = np.load("train_reg_data/inputs_reg_hct116.npy")
    y = np.load("train_reg_data/labels_reg_hct116.npy")
    x = x.transpose([0, 2, 3, 1])
    y = y.reshape((-1))
    # Creating train and development and test data
    X, X_div, Y, Y_div = train_test_split(x, y, test_size=0.33, random_state=42,shuffle=True)
    # X_div, X_test, Y_div, Y_test = train_test_split(X_div, Y_div, test_size=0.33, random_state=42)
    X_test = np.load("test_reg_data/inputs_test_reg_hct116.npy")
    X_test = X_test.transpose([0, 2, 3, 1])
    Y_test = np.load("test_reg_data/labels_test_reg_hct116.npy")
    Y_test = Y_test.reshape((-1))

    class MonitorCallback(tflearn.callbacks.Callback):
        def __init__(self):
            self.testError = []
            self.trainError = []
            self.wholeError = []
            self.testSPEAR = []
            self.test_Pval = []
            self.WSSPEAR = []
            self.WS_Pval = []
            self.trainSPEAR = []
            self.train_Pval = []

        def on_epoch_end(self, training_state):

            pred_prob = model.predict(X)
            train_SPEAR_Corrval,train_SPEAR_Pval = spearmanr(Y,pred_prob)
            train_error_val = mean_squared_error(Y,pred_prob)
            self.trainSPEAR.append( train_SPEAR_Corrval)
            self.train_Pval.append(train_SPEAR_Pval)
            self.trainError.append(train_error_val)
            print("Train Spearman Corr value :",train_SPEAR_Corrval)
            print("Train MSE value :",train_error_val)
            
            pred_prob = model.predict(X_test)
            test_error_val = mean_squared_error(Y_test,pred_prob)
            test_SPEAR_Corrval,test_SPEAR_Pval = spearmanr(Y_test,pred_prob)
            self.testSPEAR.append( test_SPEAR_Corrval)
            self.test_Pval.append(test_SPEAR_Pval)
            self.testError.append( test_error_val)
            print("Test Spearman Corr value :",test_SPEAR_Corrval)
            print("Test MSE value :",test_error_val)

            pred_prob = model.predict(x)
            ws_SPEAR_Corrval,ws_SPEAR_Pval = spearmanr( y,pred_prob)
            whole_error_val = mean_squared_error(y,pred_prob)
            self.WSSPEAR.append( ws_SPEAR_Corrval)
            self.WS_Pval.append( ws_SPEAR_Pval)
            self.wholeError.append(whole_error_val)
            print("Whole Spearman Corr value :",ws_SPEAR_Corrval)
            print("Whole MSE value :",whole_error_val)

            df=pd.DataFrame(
                {
                    'Train SPEAR CORR score':self.trainSPEAR,
                    'Train SPEAR Pval score':self.train_Pval,
                    'Test SPEAR CORR score':self.testSPEAR,
                    'Test SPEAR Pval score':self.test_Pval,
                    'Whole SPEAR CORR score':self.WSSPEAR,
                    'Whole SPEAR Pval score':self.WS_Pval,
                    'Test ERROR score':self.testError,
                    'Train ERROR score':self.trainError,
                    'Whole ERROR score':self.wholeError
                })
            df.to_csv("./TrainingOutputs/REGAccuracies_hct116-without dropout-.csv")

        def on_train_end(self,training_state):
            df=pd.DataFrame(
                {
                    'Train SPEAR CORR score':self.trainSPEAR,
                    'Train SPEAR Pval score':self.train_Pval,
                    'Test SPEAR CORR score':self.testSPEAR,
                    'Test SPEAR Pval score':self.test_Pval,
                    'Whole SPEAR CORR score':self.WSSPEAR,
                    'Whole SPEAR Pval score':self.WS_Pval,
                    'Test ERROR score':self.testError,
                    'Train ERROR score':self.trainError,
                    'Whole ERROR score':self.wholeError
                })
            df.to_csv("./TrainingOutputs/REGAccuracies_hct116-without dropout-.csv")
      
    # create model from classification
    model = create_regression_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/Classification_Model/model.tfl")

    monitorCallback = MonitorCallback()
    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch = 20, batch_size = batch_size,
    validation_set = ({'input': X_div}, {'target': Y_div}),
    snapshot_step = 1000, show_metric = True, callbacks = monitorCallback)
    
    # save the model
    model.save("./TrainingOutputs/Regression/regModel/RegressionModel.tfl")

main()