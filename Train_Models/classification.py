from Build_Models.classification import create_classification_model
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
batch_size=256



def main():
    # loading data
    X=np.load("inputs.npy")
    y=np.load("labels.npy")
    X = X.transpose([0, 2, 3, 1])
    # print(X[0])
    # print(X.shape)
    # print(y.shape)

    # Creating train and development and test data
    X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
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
    model.fit({'input': X}, {'target': Y}, n_epoch=100,batch_size=batch_size,
    validation_set=({'input': X_div}, {'target': Y_div}),
    snapshot_step=1000,show_metric=True)
    
    # save the model
    model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")

    # you can comment the tree lines above and uncomment this line so you can load your pretrained model
    # model.load("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")


    # measuring accuracy for test data
    cls_outpot = model.predict(X_test)
    for i in range(len(cls_outpot)):
        cls_outpot[i] = 1 if cls_outpot[i]>0.5 else 0
    T = np.sum(cls_outpot==Y_test)
    AccuracyMesaure= (T)/(Y_test.shape[0])
    print(AccuracyMesaure)

    # measuring accuracy for development data
    cls_outpot = model.predict(X_div)
    for i in range(len(cls_outpot)):
        cls_outpot[i] = 1 if cls_outpot[i]>0.5 else 0
    T = np.sum(cls_outpot==Y_div)
    AccuracyMesaure= (T)/(Y_div.shape[0])
    print(AccuracyMesaure)

    # measuring accuracy for train data
    cls_outpot = model.predict(X)
    for i in range(len(cls_outpot)):
        cls_outpot[i] = 1 if cls_outpot[i]>0.5 else 0
    T = np.sum(cls_outpot==Y)
    AccuracyMesaure= (T)/(Y.shape[0])
    print(AccuracyMesaure)

    # print(model.evaluate(X_test,Y_test))
    # print(model.evaluate(X_div,Y_div))
    # print(model.evaluate(X,Y))
    
main()

