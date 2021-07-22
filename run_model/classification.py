from Build_Models.classification import create_classification_model 
from Build_Models.classification_dropout import create_classification_model_with_dropout
from Build_Models.classification_attention import create_classification_model_with_attention

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

