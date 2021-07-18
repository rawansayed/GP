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


# --------------- loading the location where we will save data into --------------- #
save_location=__file__[:__file__.rindex("/")]
print(save_location)



def scenario7_hct(epochNum):
    # --------------- loading Data --------------- #
    files = ['hek293t.episgt','hela.episgt','hl60.episgt']

    dataArr_inputs_train =   np.array([None]*3)
    dataArr_inputs_test  =   np.array([None]*3)
    dataArr_labels_train =   np.array([None]*3)
    dataArr_labels_test  =   np.array([None]*3)

    # loading every piece in one big data
    for i in range(3):
        files[i]=files[i][:files[i].index('.')]
        x=np.load(f"./training_data/inputs_{files[i]}_REG.npy" )
        dataArr_inputs_train[i]    =   x
        x=np.load(f"./training_data/inputs_{files[i]}_test_REG.npy")
        dataArr_inputs_test[i]     =   x
        x=np.load(f"./training_data/labels_{files[i]}_REG.npy")
        dataArr_labels_train[i]    =   x
        x=np.load(f"./training_data/labels_{files[i]}_test_REG.npy")
        dataArr_labels_test[i]     =   x
    
    # concatente the array of 4 to get one array 

    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train))
    dataArr_inputs_test      = np.array([None])
    dataArr_labels_train     = np.concatenate((dataArr_labels_train))
    dataArr_labels_test      = np.array([None])

    # loading every piece on its own
    
    dataArr_inputs_test_hek293t=np.load(f"./training_data/inputs_hek293t_test_REG.npy")
    dataArr_labels_test_hek293t=np.load(f"./training_data/labels_hek293t_test_REG.npy")
    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train,dataArr_inputs_test_hek293t))
    dataArr_labels_train     = np.concatenate((dataArr_labels_train,dataArr_labels_test_hek293t))

    dataArr_inputs_test_hct116=np.load(f"./training_data/inputs_hct116_test_REG.npy")
    dataArr_labels_test_hct116=np.load(f"./training_data/labels_hct116_test_REG.npy")
    dataArr_inputs_train_hct116=np.load(f"./training_data/inputs_hct116_REG.npy")
    dataArr_labels_train_hct116=np.load(f"./training_data/labels_hct116_REG.npy")
    dataArr_inputs_test     = np.concatenate((dataArr_inputs_test_hct116,dataArr_inputs_train_hct116))
    dataArr_labels_test     = np.concatenate((dataArr_labels_test_hct116,dataArr_labels_train_hct116))

    dataArr_inputs_test_hl60=np.load(f"./training_data/inputs_hl60_test_REG.npy")
    dataArr_labels_test_hl60=np.load(f"./training_data/labels_hl60_test_REG.npy")
    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train,dataArr_inputs_test_hl60))
    dataArr_labels_train     = np.concatenate((dataArr_labels_train,dataArr_labels_test_hl60))
    
    dataArr_inputs_test_hela=np.load(f"./training_data/inputs_hela_test_REG.npy")
    dataArr_labels_test_hela=np.load(f"./training_data/labels_hela_test_REG.npy")
    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train,dataArr_inputs_test_hela))
    dataArr_labels_train     = np.concatenate((dataArr_labels_train,dataArr_labels_test_hela))


    # Checking the dimensions
    print("loaded data dimention check")
    print("shape of dataArr_inputs_train",dataArr_inputs_train.shape)   
    print("shape of dataArr_inputs_test",dataArr_inputs_test.shape)    
    print("shape of dataArr_labels_train",dataArr_labels_train.shape)   
    print("shape of dataArr_labels_test",dataArr_labels_test.shape)
    

    # Reshaping the arrays
    dataArr_inputs_train = dataArr_inputs_train.transpose([0, 2, 3, 1])
    dataArr_inputs_test = dataArr_inputs_test.transpose([0, 2, 3, 1])
    dataArr_inputs_test_hek293t=dataArr_inputs_test_hek293t.transpose([0, 2, 3, 1])
    dataArr_inputs_test_hct116=dataArr_inputs_test_hct116.transpose([0, 2, 3, 1])
    dataArr_inputs_test_hl60=dataArr_inputs_test_hl60.transpose([0, 2, 3, 1])
    dataArr_inputs_test_hela=dataArr_inputs_test_hela.transpose([0, 2, 3, 1])

    dataArr_labels_train=dataArr_labels_train.reshape((-1))
    dataArr_labels_test=dataArr_labels_test.reshape((-1))
    dataArr_labels_test_hek293t=dataArr_labels_test_hek293t.reshape((-1))
    dataArr_labels_test_hct116=dataArr_labels_test_hct116.reshape((-1))
    dataArr_labels_test_hl60=dataArr_labels_test_hl60.reshape((-1))
    dataArr_labels_test_hela=dataArr_labels_test_hela.reshape((-1))

    # Checking the dimensions
    print("loaded data dimensions recheck")
    print("shape of dataArr_inputs_train",dataArr_inputs_train.shape)   
    print("shape of dataArr_inputs_test",dataArr_inputs_test.shape)    
    print("shape of dataArr_labels_train",dataArr_labels_train.shape)   
    print("shape of dataArr_labels_test",dataArr_labels_test.shape)

    # Creating train and development datasets and rename the test dataset
    X_train, X_dev, Y_train, Y_dev = train_test_split(dataArr_inputs_train, dataArr_labels_train, test_size=0.33, random_state=42,shuffle=True)
    
    X_test=dataArr_inputs_test
    Y_test=dataArr_labels_test
    
    X_whole_set=np.concatenate((X_train,X_dev,X_test))
    Y_whole_set=np.concatenate((Y_train,Y_dev,Y_test))

    print("training data dimensions check")
    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_dev.shape)
    print(Y_test.shape)

    #--------------- Creating Call back function --------------- #

    # this class will compute statistical measures and save them
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
            self.loss=[]
            self.maxtestSP_first=[0]
            self.maxtestSP_second=[0]
            self.hekSP=[]
            self.hctSP=[]
            self.hlSP=[]
            self.helaSP=[]
            self.hekPVal=[]
            self.hctPVal=[]
            self.hlPVal=[]
            self.helaPVal=[]

        def on_epoch_end(self, training_state):

            pred_prob = model.predict(X_train)
            train_SPEAR_Corrval,train_SPEAR_Pval = spearmanr(Y_train,pred_prob)
            train_error_val = mean_squared_error(Y_train,pred_prob)
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

            pred_prob = model.predict(X_whole_set)
            ws_SPEAR_Corrval,ws_SPEAR_Pval = spearmanr(Y_whole_set,pred_prob)
            whole_error_val = mean_squared_error(Y_whole_set,pred_prob)
            self.WSSPEAR.append( ws_SPEAR_Corrval)
            self.WS_Pval.append( ws_SPEAR_Pval)
            self.wholeError.append(whole_error_val)
            print("Whole Spearman Corr value :",ws_SPEAR_Corrval)
            print("Whole MSE value :",whole_error_val)

            pred_prob = model.predict(dataArr_inputs_test_hek293t)
            hek_SPEAR_Corrval,hek_SPEAR_Pval = spearmanr(dataArr_labels_test_hek293t,pred_prob)
            self.hekPVal.append( hek_SPEAR_Pval)
            self.hekSP.append( hek_SPEAR_Corrval)

            pred_prob = model.predict(dataArr_inputs_test_hct116)
            hct_SPEAR_Corrval,hct_SPEAR_Pval = spearmanr(dataArr_labels_test_hct116,pred_prob)
            self.hctPVal.append( hct_SPEAR_Pval)
            self.hctSP.append( hct_SPEAR_Corrval)

            pred_prob = model.predict(dataArr_inputs_test_hl60)
            hl_SPEAR_Corrval,hl_SPEAR_Pval = spearmanr(dataArr_labels_test_hl60,pred_prob)
            self.hlPVal.append( hl_SPEAR_Pval)
            self.hlSP.append( hl_SPEAR_Corrval)

            pred_prob = model.predict(dataArr_inputs_test_hela)
            hela_SPEAR_Corrval,hela_SPEAR_Pval = spearmanr(dataArr_labels_test_hela,pred_prob)
            self.helaPVal.append( hela_SPEAR_Pval)
            self.helaSP.append( hela_SPEAR_Corrval)

            # # append loss
            self.loss.append(training_state.loss_value)
            
            
            # # Some logic to save the best model and calculate the roc curve for it
            # here the third best model with the best spearman corr we found and any overfitting
            if ((max(abs( self.maxtestSP_second))<abs(test_SPEAR_Corrval))):
                model.save(f"{save_location}/regModel/hct/BestModel/ClassificationModel.tfl")
                self.maxtestSP_second.append( test_SPEAR_Corrval)



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
                    'Whole ERROR score':self.wholeError,
                    'training loss':self.loss,
                    'hek293 spearman corr':self.hekSP,
                    # 'hct116 spearman corr':self.hctSP,
                    'hela spearman corr':self.helaSP,
                    'hl60 spearman corr':self.hlSP,
                    'hek293 P value':self.hekPVal,
                    # 'hct116 P value':self.hctPVal,
                    'hela P value':self.helaPVal,
                    'hl60 P value':self.hlPVal,
                })
            df.to_csv(f"{save_location}/statistical_measures/hct/REGAccuracies.csv")

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
                    'Whole ERROR score':self.wholeError,
                    'training loss':self.loss,
                    'hek293 spearman corr':self.hekSP,
                    # 'hct116 spearman corr':self.hctSP,
                    'hela spearman corr':self.helaSP,
                    'hl60 spearman corr':self.hlSP,
                    'hek293 P value':self.hekPVal,
                    # 'hct116 P value':self.hctPVal,
                    'hela P value':self.helaPVal,
                    'hl60 P value':self.hlPVal,
                })
            df.to_csv(f"{save_location}/statistical_measures/hct/FinalREGAccuracies.csv")

    # create model from regression
    model = create_regression_model()

    # loading encoder weights
    model.load(f"{save_location}/autoencoderModel/model.tfl")

    monitorCallback = MonitorCallback()
    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X_train}, {'target': Y_train}, n_epoch = epochNum, batch_size = batch_size,
    validation_set = ({'input': X_dev}, {'target': Y_dev}),
    snapshot_step = 1000, show_metric = True, callbacks = monitorCallback)
    
    # save the model
    model.save(f"{save_location}/regModel/hct/finalModel/RegressionModel.tfl")

scenario7_hct(2)