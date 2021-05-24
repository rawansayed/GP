from utils import Episgt
import numpy as np


# ------------- For Classification ------------- #

file_path1 = './database/paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# file_path = './examples/eg_cls_on_target.episgt'
input_data = Episgt(file_path1, num_epi_features=4, with_y=True)
x_1, y_1 = input_data.get_dataset()
X_1 = np.expand_dims(x_1, axis=2)
# X = X.transpose([0, 2, 3, 1])
np.save("inputs.npy",X_1)
np.save("labels.npy",y_1)


# --------------- For Regression --------------- #

file_path2 = './database/paper_data-regression/paper_data2/ontar/hct116.repisgt'
input_data = Episgt(file_path2, num_epi_features=4, with_y=True)
x_2, y_2 = input_data.get_dataset()
X_2 = np.expand_dims(x_2, axis=2)
np.save("inputs_reg.npy",X_2)
np.save("labels_reg.npy",y_2)