from Build_Models.regression import create_regression_model
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
import tflearn

batch_size=256

def accuracyForREGMODEL(X_pred,Y_true,lowThresh,highThresh):

    for i in range(len(X_pred)):
        X_pred[i] = 1 if X_pred[i]>=highThresh else X_pred[i]
        X_pred[i] = 0 if X_pred[i]<=lowThresh else X_pred[i]
        X_pred[i] = 2 if highThresh>X_pred[i] and X_pred[i]>lowThresh else X_pred[i]
    correctArray=X_pred==Y_true
    correct = np.sum(correctArray)
    unpridected=np.sum(X_pred==2)
    zeros=np.sum(X_pred==0)
    correctZeros=np.sum(X_pred[correctArray]==0)
    ones=np.sum(X_pred==1)
    correctOnes=np.sum(X_pred[correctArray]==1)
    AccuracyMesaure= (correct)/(Y_true.shape[0]-unpridected)

    TP=correctOnes
    TN=correctZeros
    FP=ones-correctOnes
    FN=zeros-correctZeros
    Precition=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1=2*Precition*Recall/(Precition+Recall)
    TP=TP/Y_true.shape[0]
    TN=TN/Y_true.shape[0]
    FP=FP/Y_true.shape[0]
    FN=FN/Y_true.shape[0]
    return AccuracyMesaure*100 ,TP*100 ,TN*100 ,FP*100 ,FN*100 ,Precition*100 ,Recall*100 ,F1*100 


def main():
    # loading data
    X=np.load("D:\BIOMEDICAL ENGINEERING\GP\GP\inputs_reg.npy")
    y=np.load("D:\BIOMEDICAL ENGINEERING\GP\GP\labels_reg.npy")
    X = X.transpose([0, 2, 3, 1])

    # Creating train and development and test data
    X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.1, random_state=42)

    class MonitorCallback(tflearn.callbacks.Callback):
        def __init__(self):
            self.trainAcc = []
            self.trainPrec = []
            self.trainRecall = []
            self.trainF1 = []
            self.trainTP = []
            self.trainFP = []
            self.trainTN = []
            self.trainFN = []
            self.devAcc = []
            self.devPrec = []
            self.devRecall = []
            self.devF1 = []
            self.devTP = []
            self.devFP = []
            self.devTN = []
            self.devFN = []
            self.testAcc = []
            self.testPrec = []
            self.testRecall = []
            self.testF1 = []
            self.testTP = []
            self.testFP = []
            self.testTN = []
            self.testFN = []

        def on_epoch_end(self, training_state):
            Y_pred = model.predict(X)
            trainAccuracyMesaure ,trainTP,trainTN,trainFP,trainFN,trainPrecition,trainRecall,trainF1 = accuracyForREGMODEL(Y_pred,Y,0.5,0.5)
            self.trainAcc.append( trainAccuracyMesaure)
            self.trainPrec.append(trainPrecition)
            self.trainRecall.append(trainRecall)
            self.trainF1.append(trainF1)
            self.trainTP.append( trainTP)
            self.trainTN.append( trainTN)
            self.trainFP.append( trainFP)
            self.trainFN.append( trainFN)

            Y_pred_div = model.predict(X_div)
            devAccuracyMesaure ,devTP,devTN,devFP,devFN,devPrecition,devRecall,devF1 = accuracyForREGMODEL(Y_pred_div,Y_div,0.5,0.5)
            self.devAcc.append( devAccuracyMesaure)
            self.devPrec.append( devPrecition)
            self.devRecall.append( devRecall)
            self.devF1.append( devF1)
            self.devTP.append( devTP)
            self.devTN.append( devTN)
            self.devFP.append( devFP)
            self.devFN.append( devFN)

            Y_pred_test = model.predict(X_test)
            testAccuracyMesaure ,testTP,testTN,testFP,testFN ,testPrecition,testRecall,testF1= accuracyForREGMODEL(Y_pred_test,Y_test,0.5,0.5)
            self.testAcc.append( testAccuracyMesaure)
            self.testPrec.append( testPrecition)
            self.testRecall.append( testRecall)
            self.testF1.append( testF1)
            self.testTP.append( testTP)
            self.testTN.append( testTN)
            self.testFP.append( testFP)
            self.testFN.append( testFN)

            pred_prob=model.predict(X_test)
            print("test AUC value :",roc_auc_score(Y_test, pred_prob))
            pred_prob=model.predict(x)
            print("whole AUC value :",roc_auc_score(y, pred_prob))
            

            print(trainAccuracyMesaure,
            devAccuracyMesaure,
            testAccuracyMesaure)

        def on_train_end(self,training_state):
            df=pd.DataFrame(
                {
                    'training set Accuracy': self.trainAcc,
                    'training set Precittion': self.trainPrec,
                    'training set Recall': self.trainRecall,
                    'training set F1 score': self.trainF1,
                    'development set Accuracy': self.devAcc,
                    'development set Precittion': self.devPrec,
                    'development set Recall': self.devRecall,
                    'development set F1 score': self.devF1,
                    'test set Accuracy': self.testAcc,
                    'test set Precittion': self.testPrec,
                    'test set Recall': self.testRecall,
                    'test set F1 score': self.testF1,
                    'training set True positive ratio': self.trainTP,
                    'training set True negative ratio': self.trainTN,
                    'training set False positive ratio': self.trainFP,
                    'training set False negative ratio': self.trainFN,
                    'development set True positive ratio': self.devTP,
                    'development set True negative ratio': self.devTN,
                    'development set False positive ratio': self.devFP,
                    'development set False negative ratio': self.devFN,
                    'test set True positive ratio': self.testTP,
                    'test set True negative ratio': self.testTN,
                    'test set False positive ratio': self.testFP,
                    'test set False negative ratio': self.testFN,

                })
            df.to_csv("./TrainingOutputs/REGAcuuracies.csv")
    # create model from classification
    model = create_regression_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/Regression_Model/model.tfl")

    monitorCallback = MonitorCallback()

    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=20,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True)
    
    # save the model
    model.save("./TrainingOutputs/Regression/regModel/RegressionModel.tfl")

    # measuring accuracy for test data
    low=0.1
    high =0.9
    

    Reg_outpot = model.predict(X_test)
    print(Reg_outpot)
    print(Y_test)
    test_acc = accuracyForREGMODEL(Reg_outpot,Y_test,low,high)

    # measuring accuracy for development data
    Reg_outpot = model.predict(X_div)
    dev_acc = accuracyForREGMODEL(Reg_outpot,Y_div,low,high)

    # measuring accuracy for train data
    Reg_outpot = model.predict(X)
    train_acc = accuracyForREGMODEL(Reg_outpot,Y,low,high)


    print(test_acc)
    print(dev_acc)    
    print(train_acc)

main()