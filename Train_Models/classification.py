from numpy.core.fromnumeric import reshape
import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()
from Build_Models.classification import create_classification_model
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
batch_size=256

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
    print(X_pred.reshape((-1,)),Y_true.reshape((-1,)),X_pred.shape,Y_true.shape)
    print("correct:",correct,
    "zeros:",zeros,"correct Zeros",correctZeros,
    "ones:",ones,"correct ones",correctOnes,
    "unpridected:",unpridected,"size:",Y_true.shape[0]
    ,"Accuracy:",AccuracyMesaure)
    return AccuracyMesaure


def main():
    # loading data
    X=np.load("inputs.npy")
    y=np.load("labels.npy")
    X = X.transpose([0, 2, 3, 1])
    # y=y.reshape((-1))
    # y = y.reshape([-1,1,1,1])
    # print(X[0])
    # print(X.shape)
    # print(y.shape)

    # Creating train and development and test data
    X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)
    X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.1, random_state=42)
    
    # print(X_div.shape)
    # print(Y_div.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    # Y=Y.reshape([-1,])
    # Y_test=Y_test.reshape([-1,]) 
    
    # create model from classification
    model=create_classification_model()

    # loading encoder weights
    model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")


    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=25,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True)
    
    # save the model
    model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")

    # you can comment the tree lines above and uncomment this line so you can load your pretrained model
    # model.load("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")


    # measuring accuracy for test data
    low=0.1
    high =0.9
    

    cls_outpot = model.predict(X_test)
    print(cls_outpot)
    print(Y_test)
    test_acc = accuracyForCLSMODEL(cls_outpot,Y_test,low,high)

    # measuring accuracy for development data
    cls_outpot = model.predict(X_div)
    dev_acc = accuracyForCLSMODEL(cls_outpot,Y_div,low,high)

    # measuring accuracy for train data
    cls_outpot = model.predict(X)
    train_acc = accuracyForCLSMODEL(cls_outpot,Y,low,high)


    print(test_acc)
    print(dev_acc)    
    print(train_acc)
    # print(model.evaluate(X_test,Y_test))
    # print(model.evaluate(X_div,Y_div))
    # print(model.evaluate(X,Y))
    print(model.variables)
    
main()

