
from regression import pretrained_reg_model,mean_squared_error,spearmanr,X_test,Y_test
import numpy as np
import pandas as pd
from ontarget import Episgt




path='./run_model/examples/reg.repisgt'
input_data = Episgt(path, num_epi_features=4, with_y=True)
x,y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)
x = x.transpose([0, 2, 3, 1])
model= pretrained_reg_model(enhancment="no enhancment")
y_pred=model.predict(x)
df=pd.DataFrame(
    {
        'y_predicted':y_pred,
        'true y':y,
    })
df.to_csv("./run_model/output/reg_model_output.csv")