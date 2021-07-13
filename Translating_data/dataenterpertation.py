from ontarget import Episgt
import numpy as np


# # ------------- For Classification ------------- #

# file_path1 = './database/paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# # file_path = './examples/eg_cls_on_target.episgt'
# input_data = Episgt(file_path1, num_epi_features=4, with_y=True)
# x_1, y_1 = input_data.get_dataset()
# X_1 = np.expand_dims(x_1, axis=2)
# # X = X.transpose([0, 2, 3, 1])
# np.save("inputs_cls.npy",X_1)
# np.save("labels_cls.npy",y_1)
dict = "./database/paper_data-classification/paper_data/ontar"
files = ['hct116_hart.episgt','hek293t_doench.episgt','hela_hart.episgt','hl60_xu.episgt']
totaldata_x = np.array([None]*4)
totaldata_y = np.array([None]*4)

for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata_x[i] = np.expand_dims(x, axis=2)
    totaldata_y[i] = np.expand_dims(y, axis=1)

dataArr_x = np.concatenate((totaldata_x))
print(dataArr_x.shape)
dataArr_y = np.concatenate((totaldata_y))
print(dataArr_y.shape)
np.save("inputs_cls.npy",dataArr_x)
np.save("labels_cls.npy",dataArr_y)
    

# --------------- For Regression --------------- #

# file_path2 = './database/paper_data-regression/paper_data2/ontar/hct116.repisgt'
# input_data = Episgt(file_path2, num_epi_features=4, with_y=True)
# x_2, y_2 = input_data.get_dataset()
# X_2 = np.expand_dims(x_2, axis=2)
# np.save("inputs_reg.npy",X_2)
# np.save("labels_reg.npy",y_2)


dict = "./database/paper_data-regression/paper_data2/ontar"
files = ['hct116.repisgt','hek293t.repisgt','hela.repisgt','hl60.repisgt']
totaldata_x = np.array([None]*4)
totaldata_y = np.array([None]*4)

for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata_x[i] = np.expand_dims(x, axis=2)
    totaldata_y[i] = np.expand_dims(y, axis=1)

dataArr_x = np.concatenate((totaldata_x))
print(dataArr_x.shape)
dataArr_y = np.concatenate((totaldata_y))
print(dataArr_y.shape)
np.save("inputs_reg.npy",dataArr_x)
np.save("labels_reg.npy",dataArr_y)