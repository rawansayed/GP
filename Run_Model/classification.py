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
    correct = np.sum(X_pred==Y_true)
    unpridected=np.sum(X_pred==2)
    AccuracyMesaure= (correct)/(Y_true.shape[0]-unpridected)
    print(X_pred.reshape((-1,)),Y_true.reshape((-1,)),X_pred.shape,Y_true.shape)
    print(correct,Y_true.shape[0],unpridected,AccuracyMesaure)
    return AccuracyMesaure


def main():
    # loading data
    x=np.load("inputs_cls.npy")
    y=np.load("labels_cls.npy")
    x = x.transpose([0, 2, 3, 1])
    X=np.load("./temp/X_train.npy")
    X_dev=np.load("./temp/X_dev.npy")
    X_test=np.load("./temp/X_test.npy")
    Y=np.load("./temp/y_train.npy")
    Y_dev=np.load("./temp/y_dev.npy")
    Y_test=np.load("./temp/y_test.npy")
    # y = y.reshape([-1,1,1,1])
    # print(X[0])
    # print(X.shape)
    # print(y.shape)

    # Creating train and development and test data
    # X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)
    # X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.1, random_state=42)
    
    # print(X_div.shape)
    # print(Y_div.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    # Y=Y.reshape([-1,])
    # Y_test=Y_test.reshape([-1,]) 
    
    # create model from classification
    model=create_classification_model()

    # # loading encoder weights
    # model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")


    # # start training with input as the X train data and target as Y train data
    # # and validate/develop over X_dev and Y_dev
    # model.fit({'input': X}, {'target': Y}, n_epoch=20,batch_size=batch_size,
    # validation_set=({'input': X_div}, {'target': Y_div}),
    # snapshot_step=1000,show_metric=True)
    
    # # save the model
    # model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")

    # you can comment the tree lines above and uncomment this line so you can load your pretrained model
    model.load("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")


    # measuring accuracy for test data
    

    # cls_outpot = model.predict(X_test)
    # test_acc = accuracyForCLSMODEL(cls_outpot,Y_test,0.1,0.9)

    # # measuring accuracy for development data
    # cls_outpot = model.predict(X_div)
    # dev_acc = accuracyForCLSMODEL(cls_outpot,Y_div,0.1,0.9)

    # measuring accuracy for train data
    cls_outpot = model.predict(X)
    train_acc = accuracyForCLSMODEL(cls_outpot,Y,0.5,0.5)
    cls_outpot = model.predict(X_dev)
    dev_acc = accuracyForCLSMODEL(cls_outpot,Y_dev,0.5,0.5)
    cls_outpot = model.predict(X_test)
    test_acc = accuracyForCLSMODEL(cls_outpot,Y_test,0.5,0.5)

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt

    pred_prob=model.predict(X_test)
    fpr_test, tpr_test, thresh_test = roc_curve(Y_test, pred_prob)
    print("test AUC value :",roc_auc_score(Y_test, pred_prob))
    pred_prob=model.predict(x)
    fpr_ws, tpr_ws, thresh_ws = roc_curve(y, pred_prob)
    print("whole AUC value :",roc_auc_score(y, pred_prob))

    plt.plot(fpr_test, tpr_test, linestyle='--',color='blue', label='test')
    plt.plot(fpr_ws, tpr_ws, linestyle='dotted',color='orange', label='whole set')

    plt.title('Our Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Our Model',dpi=300)
    plt.show()





    print(train_acc)
    print(dev_acc)    
    print(test_acc)
    # print(model.evaluate(X_test,Y_test))
    # print(model.evaluate(X_div,Y_div))
    # print(model.evaluate(X,Y))

    


main()

