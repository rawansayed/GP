from Build_Models.classification import create_classification_model 
from Build_Models.classification_dropout import create_classification_model_with_dropout
from Build_Models.classification_attention import create_classification_model_with_attention
import numpy as np
import matplotlib as plt
from sklearn.metrics import roc_curve,roc_auc_score



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
    np.save("./run_model/output/statistical_measures/fpr_CLS.npy",fpr)
    np.save("./run_model/output/statistical_measures/tpr_CLS.npy",tpr)
    plt.plot(fpr, tpr, linestyle='--', color='blue')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig("./run_model/output/statistical_measures/ROCforCLS",dpi=300)
    plt.close() 



# --------------- loading test Data --------------- #
files = ['hek293t_doench.episgt','hct116_hart.episgt','hl60_xu.episgt','hela_hart.episgt']

dataArr_inputs_test  =   np.array([None]*4)
dataArr_labels_test  =   np.array([None]*4)

# loading every piece in one big data
for i in range(4):
    files[i]=files[i][:files[i].index('.')]
    x=np.load(f"./training_data/inputs_{files[i]}_test_CLS.npy")
    dataArr_inputs_test[i]     =   x
    x=np.load(f"./training_data/labels_{files[i]}_test_CLS.npy")
    dataArr_labels_test[i]     =   x
# concatenate and reshape
dataArr_inputs_test      = np.concatenate((dataArr_inputs_test))
dataArr_labels_test      = np.concatenate((dataArr_labels_test))
dataArr_inputs_test = dataArr_inputs_test.transpose([0, 2, 3, 1])
dataArr_labels_test=dataArr_labels_test.reshape((-1))
X_test=dataArr_inputs_test
Y_test=dataArr_labels_test


def pretrained_cls_model(enhancment="no enhancment"):
    

    # create model from classification
    if enhancment == "no enhancment":
        model=create_classification_model()
        model.load("./scenarios/scenario3/clsModel/secondBestModel/ClassificationModel.tfl")
    if enhancment == "dropout":
        model=create_classification_model_with_dropout()
        model.load("./enhancments/dropout_cls/clsModel/secondBestModel/ClassificationModel.tfl")
    if enhancment == "attention":
        model=create_classification_model_with_attention()
        model.load("./enhancments/attention_cls/clsModel/BestModel/ClassificationModel.tfl")

    return model


"""
for classification with attention
"./enhancments/attention_cls/clsModel/BestModel/ClassificationModel.tfl"
test Accuracy value : 67.66109785202865
test AUC value : 0.6642642154867395

"./enhancments/attention_cls/clsModel/secondBestModel/ClassificationModel.tfl"
test Accuracy value : 67.3926014319809
test AUC value : 0.6648832075648934

"./enhancments/attention_cls/clsModel/thirdBestModel/ClassificationModel.tfl"
test Accuracy value : 67.3926014319809
test AUC value : 0.6648832075648934

"./enhancments/attention_cls/clsModel/finalModel/ClassificationModel.tfl"
test Accuracy value : 67.6909307875895
test AUC value : 0.6546916585778251

"""


"""
for classification with dropout
"./enhancments/dropout_cls/clsModel/BestModel/ClassificationModel.tfl"
test Accuracy value : 67.33293556085918
test AUC value : 0.7090882049034035

"./enhancments/dropout_cls/clsModel/secondBestModel/ClassificationModel.tfl"
test Accuracy value : 67.33293556085918
test AUC value : 0.7090882049034035

"./enhancments/dropout_cls/clsModel/thirdBestModel/ClassificationModel.tfl"
test Accuracy value : 67.33293556085918
test AUC value : 0.7090882049034035

"./enhancments/dropout_cls/clsModel/finalModel/ClassificationModel.tfl"
test Accuracy value : 67.33293556085918
test AUC value : 0.7090882049034035

"""


"""
for normal classification with no enhancment
# "./scenarios/scenario1/clsModel/BestModel/ClassificationModel.tfl"
# test Accuracy value : 64.28997613365155
# test AUC value : 0.6673732661176692

# "./scenarios/scenario1/clsModel/secondBestModel/ClassificationModel.tfl"
# test Accuracy value : 67.57159904534606
# test AUC value : 0.6974780427090834

# "./scenarios/scenario1/clsModel/thirdBestModel/ClassificationModel.tfl" 
# test Accuracy value : 69.39140811455847
# test AUC value : 0.7133174296270783

# "./scenarios/scenario1/clsModel/finalModel/ClassificationModel.tfl"
# test Accuracy value : 68.43675417661098
# test AUC value : 0.6925094717725522

# "./scenarios/scenario2/clsModel/BestModel/ClassificationModel.tfl"
test Accuracy value : 63.782816229116946
test AUC value : 0.6875336600181607

# "./scenarios/scenario2/clsModel/secondBestModel/ClassificationModel.tfl"
test Accuracy value : 66.52744630071598
test AUC value : 0.71160389986536

# "./scenarios/scenario2/clsModel/thirdBestModel/ClassificationModel.tfl" 
test Accuracy value : 68.4964200477327
test AUC value : 0.7134724222688418

# "./scenarios/scenario2/clsModel/finalModel/ClassificationModel.tfl"
test Accuracy value : 69.27207637231504
test AUC value : 0.7052981259980586

# "./scenarios/scenario3/clsModel/BestModel/ClassificationModel.tfl"
test Accuracy value : 65.90095465393794
test AUC value : 0.6829046247299371

# "./scenarios/scenario3/clsModel/secondBestModel/ClassificationModel.tfl"
test Accuracy value : 71.09188544152745
test AUC value : 0.7009694868021417

# "./scenarios/scenario3/clsModel/thirdBestModel/ClassificationModel.tfl" 
test Accuracy value : 71.09188544152745
test AUC value : 0.7009694868021417

# "./scenarios/scenario3/clsModel/finalModel/ClassificationModel.tfl"
test Accuracy value : 70.4653937947494
test AUC value : 0.6996001894354511

# "./scenarios/scenario4/clsModel/hct/BestModel/ClassificationModel.tfl"
test Accuracy value : 80.75775656324582
test AUC value : 0.8195883692895388

# "./scenarios/scenario4/clsModel/hct/finalModel/ClassificationModel.tfl"
test Accuracy value : 81.47374701670644
test AUC value : 0.81963983780568

# "./scenarios/scenario4/clsModel/hek/BestModel/ClassificationModel.tfl"
test Accuracy value : 64.23031026252983
test AUC value : 0.6675159297992924

# "./scenarios/scenario4/clsModel/hek/finalModel/ClassificationModel.tfl"
test Accuracy value : 80.8472553699284
test AUC value : 0.8137368099696277

# "./scenarios/scenario4/clsModel/hela/BestModel/ClassificationModel.tfl"
test Accuracy value : 77.95346062052506
test AUC value : 0.7720040626859129

# "./scenarios/scenario4/clsModel/hela/finalModel/ClassificationModel.tfl"
test Accuracy value : 77.95346062052506
test AUC value : 0.7720040626859129

# "./scenarios/scenario4/clsModel/hl/BestModel/ClassificationModel.tfl"
test Accuracy value : 71.9272076372315
test AUC value : 0.7256653724520149

# "./scenarios/scenario4/clsModel/hl/finalModel/ClassificationModel.tfl"
test Accuracy value : 80.36992840095465
test AUC value : 0.8258205607915584

"""

