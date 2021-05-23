from ontarget import Episgt
import numpy as np

dict = "./database/paper_data-classification/paper_data/ontar"
files = ['hct116_hart.episgt','hek293t_doench.episgt','hela_hart.episgt','hl60_xu.episgt']
totaldata = np.array([None]*4)
for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata[i] = np.expand_dims(x, axis=2)
    
dataArr = np.concatenate((totaldata))
print(dataArr.shape)
np.save("autoencoder_inputs.npy",dataArr)


# file_path1 = './database/paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# input_data = Episgt(file_path1, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X1 = np.expand_dims(x, axis=2)
# # X = X.transpose([0, 2, 3, 1])
# # print(X.shape)



# file_path2 = './database/paper_data-classification/paper_data/ontar/hek293t_doench.episgt'
# input_data = Episgt(file_path2, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X2 = np.expand_dims(x, axis=2)
# # print(X.shape)



# file_path3 = './database/paper_data-classification/paper_data/ontar/hela_hart.episgt'
# input_data = Episgt(file_path3, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X3 = np.expand_dims(x, axis=2)
# # print(X.shape)

# file_path4 = './database/paper_data-classification/paper_data/ontar/hl60_xu.episgt'
# input_data = Episgt(file_path4, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# X4 = np.expand_dims(x, axis=2)



# print(alldata)
