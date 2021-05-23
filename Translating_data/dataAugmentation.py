from utils import Episgt
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
datainverse=dataArr[...,::-1]
print(dataArr.shape)
np.save("sav1.npy",dataArr)
print(dataArr[0])

print(datainverse.shape)
np.save("sav2.npy",datainverse)
print(datainverse[0])

Data=np.concatenate((datainverse,dataArr))
print(Data.shape)
np.save("DATA.npy",Data)