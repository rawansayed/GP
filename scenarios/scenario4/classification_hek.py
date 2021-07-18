from numpy.core.fromnumeric import reshape
from numpy.lib.npyio import save
import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()
import tflearn
from Build_Models.classification import create_classification_model
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
batch_size=256
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import os

# --------------- loading the location where we will save data into --------------- #
save_location=__file__[:__file__.rindex("/")]
print(save_location)

# --------------- creating a function to measure statistical measurments --------------- #
def accuracyForCLSMODEL(X_pred,Y_true,lowThresh,highThresh):

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
    # print(X_pred.reshape((-1,)),Y_true.reshape((-1,)),X_pred.shape,Y_true.shape)
    # print("correct:",correct,
    # "zeros:",zeros,"correct Zeros",correctZeros,
    # "ones:",ones,"correct ones",correctOnes,
    # "unpridected:",unpridected,"size:",Y_true.shape[0]
    # ,"Accuracy:",AccuracyMesaure)
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

# --------------- creating a function to save roc curve for every best model --------------- #
def createROCCurve(model,X_test,Y_test,model_name):
    PredProb=model.predict(X_test)
    fpr,tpr, thresh1 =roc_curve(Y_test,PredProb)
    np.save(f"{save_location}/statistical_measures/hek/fpr_{model_name}.npy",fpr)
    np.save(f"{save_location}/statistical_measures/hek/tpr_{model_name}.npy",tpr)
    plt.plot(fpr, tpr, linestyle='--', color='blue')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f"{save_location}/statistical_measures/hek/ROCfor{model_name}",dpi=300)
    plt.close() 


def scenario4_hek(epochNum):

    # --------------- loading Data --------------- #
    files = ['hct116_hart.episgt','hl60_xu.episgt','hela_hart.episgt']

    dataArr_inputs_train =   np.array([None]*3)
    dataArr_inputs_test  =   np.array([None]*3)
    dataArr_labels_train =   np.array([None]*3)
    dataArr_labels_test  =   np.array([None]*3)

    # loading every piece in one big data
    for i in range(3):
        files[i]=files[i][:files[i].index('.')]
        x=np.load(f"./training_data/inputs_{files[i]}_CLS.npy" )
        dataArr_inputs_train[i]    =   x
        x=np.load(f"./training_data/inputs_{files[i]}_test_CLS.npy")
        dataArr_inputs_test[i]     =   x
        x=np.load(f"./training_data/labels_{files[i]}_CLS.npy")
        dataArr_labels_train[i]    =   x
        x=np.load(f"./training_data/labels_{files[i]}_test_CLS.npy")
        dataArr_labels_test[i]     =   x
    
    # concatente the array of 4 to get one array 

    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train))
    dataArr_inputs_test      = np.array([None])
    dataArr_labels_train     = np.concatenate((dataArr_labels_train))
    dataArr_labels_test      = np.array([None])

    # loading every piece on its own
    
    dataArr_inputs_train_hek293t=np.load(f"./training_data/inputs_hek293t_doench_CLS.npy")
    dataArr_labels_train_hek293t=np.load(f"./training_data/labels_hek293t_doench_CLS.npy")
    dataArr_inputs_test_hek293t=np.load(f"./training_data/inputs_hek293t_doench_test_CLS.npy")
    dataArr_labels_test_hek293t=np.load(f"./training_data/labels_hek293t_doench_test_CLS.npy")
    dataArr_inputs_test     = np.concatenate((dataArr_inputs_train_hek293t,dataArr_inputs_test_hek293t))
    dataArr_labels_test     = np.concatenate((dataArr_labels_train_hek293t,dataArr_labels_test_hek293t))

    dataArr_inputs_test_hct116=np.load(f"./training_data/inputs_hct116_hart_test_CLS.npy")
    dataArr_labels_test_hct116=np.load(f"./training_data/labels_hct116_hart_test_CLS.npy")
    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train,dataArr_inputs_test_hct116))
    dataArr_labels_train     = np.concatenate((dataArr_labels_train,dataArr_labels_test_hct116))

    dataArr_inputs_test_hl60=np.load(f"./training_data/inputs_hl60_xu_test_CLS.npy")
    dataArr_labels_test_hl60=np.load(f"./training_data/labels_hl60_xu_test_CLS.npy")
    dataArr_inputs_train     = np.concatenate((dataArr_inputs_train,dataArr_inputs_test_hl60))
    dataArr_labels_train     = np.concatenate((dataArr_labels_train,dataArr_labels_test_hl60))
    
    dataArr_inputs_test_hela=np.load(f"./training_data/inputs_hela_hart_test_CLS.npy")
    dataArr_labels_test_hela=np.load(f"./training_data/labels_hela_hart_test_CLS.npy")
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
    X_train, X_dev, Y_train, Y_dev = train_test_split(dataArr_inputs_train, dataArr_labels_train, test_size=0.30, random_state=42,shuffle=True)
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
    
    # --------------- Creating Call back function --------------- #

    # this class will compute statistical measures and save them
    class MonitorCallback(tflearn.callbacks.Callback):
        def __init__(self):
            # initialize arrays that will save statistical measuresevery epoch
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
            self.testAUC = []
            self.WSAUC = []
            self.maxtestAUC_first=[0]
            self.maxtestAUC_second=[0]
            self.maxtestAUC_third=[0]
            self.loss=[]
            self.hekAUC=[]
            self.hctAUC=[]
            self.hlAUC=[]
            self.helaAUC=[]


        def on_epoch_end(self, training_state):

            # compute statistical measures for X_train X_dev and X_test
            Y_pred = model.predict(X_train)
            trainAccuracyMesaure ,trainTP,trainTN,trainFP,trainFN,trainPrecition,trainRecall,trainF1 = accuracyForCLSMODEL(Y_pred,Y_train,0.5,0.5)
            self.trainAcc.append( trainAccuracyMesaure)
            self.trainPrec.append(trainPrecition)
            self.trainRecall.append(trainRecall)
            self.trainF1.append(trainF1)
            self.trainTP.append( trainTP)
            self.trainTN.append( trainTN)
            self.trainFP.append( trainFP)
            self.trainFN.append( trainFN)

            Y_pred_div = model.predict(X_dev)
            devAccuracyMesaure ,devTP,devTN,devFP,devFN,devPrecition,devRecall,devF1 = accuracyForCLSMODEL(Y_pred_div,Y_dev,0.5,0.5)
            self.devAcc.append( devAccuracyMesaure)
            self.devPrec.append( devPrecition)
            self.devRecall.append( devRecall)
            self.devF1.append( devF1)
            self.devTP.append( devTP)
            self.devTN.append( devTN)
            self.devFP.append( devFP)
            self.devFN.append( devFN)

            Y_pred_test = model.predict(X_test)
            testAccuracyMesaure ,testTP,testTN,testFP,testFN ,testPrecition,testRecall,testF1= accuracyForCLSMODEL(Y_pred_test,Y_test,0.5,0.5)
            self.testAcc.append( testAccuracyMesaure)
            self.testPrec.append( testPrecition)
            self.testRecall.append( testRecall)
            self.testF1.append( testF1)
            self.testTP.append( testTP)
            self.testTN.append( testTN)
            self.testFP.append( testFP)
            self.testFN.append( testFN)

            # Compute AUC value for X_test and the whole set

            pred_prob=model.predict(X_test)
            test_auc_val=roc_auc_score(Y_test, pred_prob)
            print("test AUC value :",test_auc_val)
            self.testAUC.append( test_auc_val)

            pred_prob=model.predict(X_whole_set)
            ws_auc_val=roc_auc_score(Y_whole_set, pred_prob)
            print("whole AUC value :",ws_auc_val)
            self.WSAUC.append( ws_auc_val)

            pred_prob=model.predict(dataArr_inputs_test_hek293t)
            hek_auc_val=roc_auc_score(dataArr_labels_test_hek293t, pred_prob)
            self.hekAUC.append( hek_auc_val)

            pred_prob=model.predict(dataArr_inputs_test_hct116)
            hct_auc_val=roc_auc_score(dataArr_labels_test_hct116, pred_prob)
            self.hctAUC.append( hct_auc_val)

            pred_prob=model.predict(dataArr_inputs_test_hl60)
            hl_auc_val=roc_auc_score(dataArr_labels_test_hl60, pred_prob)
            self.hlAUC.append( hl_auc_val)

            pred_prob=model.predict(dataArr_inputs_test_hela)
            hela_auc_val=roc_auc_score(dataArr_labels_test_hela, pred_prob)
            self.helaAUC.append( hela_auc_val)

            print(
                "train Acc",trainAccuracyMesaure,
                "dev Acc",devAccuracyMesaure,
                "test Acc",testAccuracyMesaure
            )
            # # append loss
            self.loss.append(training_state.loss_value)

            # # Some logic to save the best model and calculate the roc curve for it
            # here the best model with the best AUC we found 
            if ((max( self.maxtestAUC_first)<test_auc_val)):
                model.save(f"{save_location}/clsModel/hek/BestModel/ClassificationModel.tfl")
                self.maxtestAUC_first.append( test_auc_val)
                createROCCurve(model,X_test,Y_test,"hekBestModel")
                
            
            # saving the important statistical measures every epoch
            df=pd.DataFrame(
                {
                    'training set Accuracy': self.trainAcc,
                    # 'training set Precittion': self.trainPrec,
                    # 'training set Recall': self.trainRecall,
                    # 'training set F1 score': self.trainF1,
                    'development set Accuracy': self.devAcc,
                    # 'development set Precittion': self.devPrec,
                    # 'development set Recall': self.devRecall,
                    # 'development set F1 score': self.devF1,
                    'test set Accuracy': self.testAcc,
                    # 'test set Precittion': self.testPrec,
                    # 'test set Recall': self.testRecall,
                    # 'test set F1 score': self.testF1,
                    # 'training set True positive ratio': self.trainTP,
                    # 'training set True negative ratio': self.trainTN,
                    # 'training set False positive ratio': self.trainFP,
                    # 'training set False negative ratio': self.trainFN,
                    # 'development set True positive ratio': self.devTP,
                    # 'development set True negative ratio': self.devTN,
                    # 'development set False positive ratio': self.devFP,
                    # 'development set False negative ratio': self.devFN,
                    'test set True positive ratio': self.testTP,
                    'test set True negative ratio': self.testTN,
                    # 'test set False positive ratio': self.testFP,
                    # 'test set False negative ratio': self.testFN,
                    'test AUC score':self.testAUC,
                    # 'whole AUC score':self.WSAUC,
                    'loss':self.loss,

                })
            df.to_csv(f"{save_location}/statistical_measures/hek/ClsAccuracies.csv")

        def on_train_end(self,training_state):
            print(self.loss.__len__())
            print(self.hlAUC.__len__())
            # saving all statistical measures when training ends
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
                    'test AUC score':self.testAUC,
                    'whole AUC score':self.WSAUC,
                    'loss':self.loss,
                    # 'hek293 AUC score':self.hekAUC,
                    'hct116 AUC score':self.hctAUC,
                    'hela AUC score':self.helaAUC,
                    'hl60 AUC score':self.hlAUC,
                })
            df.to_csv(f"{save_location}/statistical_measures/hek/ClsAccuraciesFinal.csv")

            

    # create model from classification
    
    model=create_classification_model()


    # load auto encoder model
    model.load(f"{save_location}/autoencoderModel/model.tfl")

    monitorCallback = MonitorCallback()

    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X_train}, {'target': Y_train}, n_epoch=epochNum,batch_size=batch_size,
    validation_set=({'input': X_dev}, {'target': Y_dev}),
    snapshot_step=1000,show_metric=True, callbacks=monitorCallback)
    
    # save the last model
    model.save(f"{save_location}/clsModel/hek/finalModel/ClassificationModel.tfl")


    # measuring accuracy for test data
    low=0.1
    high =0.9
    

    cls_outpot = model.predict(X_test)
    test_acc = accuracyForCLSMODEL(cls_outpot,Y_test,low,high)

    # measuring accuracy for development data
    cls_outpot = model.predict(X_dev)
    dev_acc = accuracyForCLSMODEL(cls_outpot,Y_dev,low,high)

    # measuring accuracy for train data
    cls_outpot = model.predict(X_train)
    train_acc = accuracyForCLSMODEL(cls_outpot,Y_train,low,high)


    print(
        "train Acc",train_acc,
        "dev Acc",dev_acc,
        "test Acc",test_acc
    )


scenario4_hek(2)

