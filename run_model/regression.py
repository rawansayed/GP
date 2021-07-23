from Build_Models.regression import create_regression_model 
from Build_Models.regression_dropout import create_regression_model_with_dropout
from Build_Models.regression_attention import create_regression_model_with_attention
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np
 # --------------- loading Data --------------- #
files = ['hek293t.episgt','hela.episgt','hct116.episgt','hl60.episgt']
dataArr_inputs_test  =   np.array([None]*4)
dataArr_labels_test  =   np.array([None]*4)
# loading every piece in one big data
for i in range(4):
    files[i]=files[i][:files[i].index('.')]
    x=np.load(f"./training_data/inputs_{files[i]}_test_REG.npy")
    dataArr_inputs_test[i]     =   x
    x=np.load(f"./training_data/labels_{files[i]}_test_REG.npy")
    dataArr_labels_test[i]     =   x
# concatente the array of 4 to get one array 
dataArr_inputs_test      = np.concatenate((dataArr_inputs_test))
dataArr_labels_test      = np.concatenate((dataArr_labels_test))
dataArr_inputs_test = dataArr_inputs_test.transpose([0, 2, 3, 1])
dataArr_labels_test=dataArr_labels_test.reshape((-1))
X_test=dataArr_inputs_test
Y_test=dataArr_labels_test










def pretrained_reg_model(enhancment="no enhancment"):
    

    # create model from classification
    if enhancment == "no enhancment":
        model=create_regression_model()
        model.load("./scenarios/scenario6/regModel/thirdBestModel/ClassificationModel.tfl")
    if enhancment == "dropout":
        model=create_regression_model_with_dropout()
        model.load("./enhancments/dropout_reg/regModel/thirdBestModel/ClassificationModel.tfl")
    if enhancment == "attention":
        model=create_regression_model_with_attention()
        model.load("./enhancments/attention_reg/regModel/thirdBestModel/ClassificationModel.tfl")

    return model


"""
for regression model with attention

"./enhancments/attention_reg/regModel/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.12630249750314507
Test Pvalue : 5.560799899549878e-15
Test MSE value : 0.029209616

"./enhancments/attention_reg/regModel/thirdBestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.19720300464754606
Test Pvalue : 1.288415447470461e-34
Test MSE value : 0.029820437

"./enhancments/attention_reg/regModel/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.21434419407451152
Test Pvalue : 1.0004915046479287e-40
Test MSE value : 0.032414112

"""



"""
for regression model with dropout

"./enhancments/dropout_reg/regModel/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.3089411745351265
Test Pvalue : 8.191531055516354e-85
Test MSE value : 0.024200544

"./enhancments/dropout_reg/regModel/thirdBestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.3089411745351265
Test Pvalue : 8.191531055516354e-85
Test MSE value : 0.024200544

"./enhancments/dropout_reg/regModel/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.2896430857310438
Test Pvalue : 2.534234818621121e-74
Test MSE value : 0.025018202

"""



"""
for regression model with no enhancments
"./scenarios/scenario6/regModel/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.2214713955039721
Test Pvalue : 1.983400816065288e-43
Test MSE value : 0.06687444

"./scenarios/scenario6/regModel/thirdBestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.3952939849155957
Test Pvalue : 2.6568130457908383e-142
Test MSE value : 0.02409169

"./scenarios/scenario6/regModel/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.4087671290061256
Test Pvalue : 5.688987942594142e-153
Test MSE value : 0.02484151

# "./scenarios/scenario7/regModel/hct/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.7177728812187373
Test Pvalue : 0.0
Test MSE value : 0.013914336

# "./scenarios/scenario7/regModel/hct/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.7073092061277186
Test Pvalue : 0.0
Test MSE value : 0.016922968

# "./scenarios/scenario7/regModel/hek/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.6365298917983627
Test Pvalue : 0.0
Test MSE value : 0.014217936

# "./scenarios/scenario7/regModel/hek/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.6365298917983627
Test Pvalue : 0.0
Test MSE value : 0.014217936

# "./scenarios/scenario7/regModel/hela/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.5719276598714741
Test Pvalue : 0.0
Test MSE value : 0.024042983

# "./scenarios/scenario7/regModel/hela/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.5195469710949443
Test Pvalue : 9.88173754557735e-262
Test MSE value : 0.025831336

# "./scenarios/scenario7/regModel/hl/BestModel/ClassificationModel.tfl"
Test Spearman Corr value : 0.2991578251196238
Test Pvalue : 2.149227472353362e-79
Test MSE value : 0.029789122

# "./scenarios/scenario7/regModel/hl/finalModel/RegressionModel.tfl"
Test Spearman Corr value : 0.7113122494700089
Test Pvalue : 0.0
Test MSE value : 0.026292099

"""








