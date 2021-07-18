from ontarget import Episgt
import numpy as np
from sklearn.model_selection import train_test_split

# # ------------- For Auto-encoder ------------- #

dict = "./database/paper_data-classification/paper_data/ontar"
files = ['hct116_hart.episgt','hek293t_doench.episgt','hela_hart.episgt','hl60_xu.episgt']
totaldata = np.array([None]*4)
for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata[i] = np.expand_dims(x, axis=2)
    
dataArr_CLS = np.concatenate((totaldata))
print("shape of dataArr_CLS",dataArr_CLS.shape)

dict = "./database/paper_data-regression/paper_data2/ontar"
files = ['hct116.repisgt','hek293t.repisgt','hela.repisgt','hl60.repisgt']
totaldata = np.array([None]*4)
for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata[i] = np.expand_dims(x, axis=2)
    
dataArr_REG = np.concatenate((totaldata))
print("shape of dataArr_REG",dataArr_REG.shape)

Data_encoder = np.concatenate((dataArr_REG,dataArr_CLS))
print("shape of Data_encoder",Data_encoder.shape)
np.save("./training_data/DATA_AutoEncoder.npy",Data_encoder)

# # ------------- For Auto-encoder augmintation ------------- #
Data_encoder_inverse=Data_encoder[...,::-1]

print("shape of Data_encoder",Data_encoder.shape)
print("shape of Data_encoder_inverse",Data_encoder_inverse.shape)
print("first element of Data_encoder")
print(Data_encoder[0])
print("first element of Data_encoder_inverse")
print(Data_encoder_inverse[0])

Data_aug = np.concatenate((Data_encoder_inverse,Data_encoder))
print("shape of Data_aug",Data_aug.shape)
np.save("./training_data/DATA_AutoEncoder_Aug.npy",Data_aug)

# # ------------- For Classification ------------- #

dict = "./database/paper_data-classification/paper_data/ontar"
files = ['hek293t_doench.episgt','hct116_hart.episgt','hl60_xu.episgt','hela_hart.episgt']
totaldata_inputs     =   np.array([None])
totaldata_labels     =   np.array([None])
dataArr_inputs_train =   np.array([None])
dataArr_inputs_test  =   np.array([None])
dataArr_labels_train =   np.array([None])
dataArr_labels_test  =   np.array([None])

for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    
    totaldata_inputs = np.expand_dims(x, axis=2)
    totaldata_labels = np.expand_dims(y, axis=1)
    
    dataArr_inputs_train ,dataArr_inputs_test ,dataArr_labels_train ,dataArr_labels_test = train_test_split(
    totaldata_inputs, totaldata_labels, test_size=0.20, random_state=42,shuffle=True)
    
    files[i]=files[i][:files[i].index('.')]
    print(f"shape of inputs_{files[i]}_CLS",dataArr_inputs_train.shape)
    print(f"shape of inputs_{files[i]}_test_CLS",dataArr_inputs_test.shape)
    print(f"shape of labels_{files[i]}_CLS",dataArr_labels_train.shape)
    print(f"shape of labels_{files[i]}_test_CLS",dataArr_labels_test.shape)

    np.save(f"./training_data/inputs_{files[i]}_CLS.npy"        ,dataArr_inputs_train   )
    np.save(f"./training_data/inputs_{files[i]}_test_CLS.npy"   ,dataArr_inputs_test   )
    
    np.save(f"./training_data/labels_{files[i]}_CLS.npy"        ,dataArr_labels_train    )
    np.save(f"./training_data/labels_{files[i]}_test_CLS.npy"   ,dataArr_labels_test    )


# # ------------- For Regression ------------- #

dict    =   "./database/paper_data-regression/paper_data2/ontar"
files   =   ['hek293t.repisgt','hela.repisgt','hct116.repisgt','hl60.repisgt']
totaldata_inputs     =   np.array([None])
totaldata_labels     =   np.array([None])
dataArr_inputs_train =   np.array([None])
dataArr_inputs_test  =   np.array([None])
dataArr_labels_train =   np.array([None])
dataArr_labels_test  =   np.array([None])

for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata_inputs = np.expand_dims(x, axis=2)
    totaldata_labels = np.expand_dims(y, axis=1)
    dataArr_inputs_train ,dataArr_inputs_test ,dataArr_labels_train ,dataArr_labels_test = train_test_split(
    totaldata_inputs, totaldata_labels, test_size=0.20, random_state=42,shuffle=True)
    
    files[i]=files[i][:files[i].index('.')]

    print(f"shape of inputs_{files[i]}_REG",dataArr_inputs_train.shape)
    print(f"shape of inputs_{files[i]}_test_REG",dataArr_inputs_test.shape)
    print(f"shape of labels_{files[i]}_REG",dataArr_labels_train.shape)
    print(f"shape of labels_{files[i]}_test_REG",dataArr_labels_test.shape)

    np.save(f"./training_data/inputs_{files[i]}_REG.npy"        ,dataArr_inputs_train   )
    np.save(f"./training_data/inputs_{files[i]}_test_REG.npy"   ,dataArr_inputs_test   )
    
    np.save(f"./training_data/labels_{files[i]}_REG.npy"        ,dataArr_labels_train    )
    np.save(f"./training_data/labels_{files[i]}_test_REG.npy"   ,dataArr_labels_test    )

    

# --------------- For Regression --------------- #

# file_path2 = './database/paper_data-regression/paper_data2/ontar/hct116.repisgt'
# input_data = Episgt(file_path2, num_epi_features=4, with_y=True)
# x_2, y_2 = input_data.get_dataset()
# X_2 = np.expand_dims(x_2, axis=2)
# np.save("inputs_reg.npy", X_2)
# np.save("labels_reg.npy", y_2)


# !!!! 3 Cell lines Only !!!!

# dict = "./database/paper_data-regression/paper_data2/ontar"
# files = ['hek293t.repisgt','hela.repisgt','hct116.repisgt']
# totaldata_x = np.array([None]*3)
# totaldata_y = np.array([None]*3)

# for i in range(3):
#     path = dict+'/'+files[i]
#     input_data = Episgt(path, num_epi_features=4, with_y=True)
#     x, y = input_data.get_dataset()
#     totaldata_x[i] = np.expand_dims(x, axis=2)
#     totaldata_y[i] = np.expand_dims(y, axis=1)

# dataArr_x = np.concatenate((totaldata_x))
# print(dataArr_x.shape)
# dataArr_y = np.concatenate((totaldata_y))
# print(dataArr_y.shape)
# np.save("train_reg_data/inputs_reg_hl60.npy", dataArr_x)
# np.save("train_reg_data/labels_reg_hl60.npy", dataArr_y)


# ----------------------- Test ---------------------

# dict = "./database/paper_data-regression/paper_data2/ontar"
# files = ['hek293t.repisgt']
# totaldata_x = np.array([None])
# totaldata_y = np.array([None])

# for i in range(1):
#     path = dict + '/' + files[i]
#     input_data = Episgt(path, num_epi_features = 4, with_y = True)
#     x, y = input_data.get_dataset()
#     totaldata_x[i] = np.expand_dims(x, axis=2)
#     totaldata_y[i] = np.expand_dims(y, axis=1)
    
# dataArr_x = np.concatenate((totaldata_x))
# dataArr_y = np.concatenate((totaldata_y))
# print(dataArr_x.shape)
# print(dataArr_y.shape)

# np.save("test_reg_data/inputs_test_reg_hct116.npy", dataArr_x)
# np.save("test_reg_data/labels_test_reg_hct116.npy", dataArr_y)