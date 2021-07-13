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
    x=np.load("inputs_reg.npy")
    y=np.load("labels_reg.npy")
    x = x.transpose([0, 2, 3, 1])
    y = y.reshape((-1))
    # Creating train and development and test data
    X, X_div, Y, Y_div = train_test_split(x, y, test_size=0.33, random_state=42,shuffle=True)
    # X_div, X_test, Y_div, Y_test = train_test_split(X_div, Y_div, test_size=0.33, random_state=42)
    X_test=np.load('./inputs_test_reg.npy')
    X_test = X_test.transpose([0, 2, 3, 1])
    Y_test=np.load('./labels_test_reg.npy')
    Y_test=Y_test.reshape((-1))

    class MonitorCallback(tflearn.callbacks.Callback):
        def __init__(self):
            self.testError =[]
            self.trainError =[]
            self.testSPEAR = []
            self.WSSPEAR = []
     
        def on_epoch_end(self, training_state):
            
            pred_prob=model.predict(X_test)
            print("test Spearman Corr value :",spearmanr(Y_test,pred_prob))
            test_error_val=mean_squared_error(Y_test,pred_prob)
            print("Test MSE value :",test_error_val)
            test_SPEAR_val=spearmanr(Y_test,pred_prob)
            self.testSPEAR.append( test_SPEAR_val)
            self.testError.append( test_error_val)

            pred_prob=model.predict(x)
            print("Whole Spearman Corr value :",spearmanr(y,pred_prob))
            ws_SPEAR_val=spearmanr( y,pred_prob)
            self.WSSPEAR.append( ws_SPEAR_val)

            pred_prob=model.predict(X)
            train_error_val=mean_squared_error(Y,pred_prob)
            print("Train MSE value :",train_error_val)
            self.trainError.append(train_error_val)
            df=pd.DataFrame(
                {
                    'test SPEAR score':self.testSPEAR,
                    'whole SPEAR score':self.WSSPEAR,
                    'test ERROR score':self.testError,
                    'train ERROR score':self.trainError,
                })
            df.to_csv("./TrainingOutputs/REGAccuracies.csv")

        def on_train_end(self,training_state):
            df=pd.DataFrame(
                {
                    'test SPEAR score':self.testSPEAR,
                    'whole SPEAR score':self.WSSPEAR,
                    'test ERROR score':self.testError,
                    'train ERROR score':self.trainError,
                })
            df.to_csv("./TrainingOutputs/REGAccuracies.csv")
      
    # create model from classification
    model = create_regression_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/Classification_Model/model.tfl")

    monitorCallback = MonitorCallback()
    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=20,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True,callbacks=monitorCallback)
    
    # save the model
    model.save("./TrainingOutputs/Regression/regModel/RegressionModel.tfl")

main()