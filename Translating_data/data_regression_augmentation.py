from ontarget import Episgt
import numpy as np

dict = "./database/paper_data-regression/paper_data2/ontar"
files = ['hct116.repisgt','hek293t.repisgt','hela.repisgt','hl60.repisgt']
totaldata2 = np.array([None]*4)
for i in range(4):
    path = dict+'/'+files[i]
    input_data = Episgt(path, num_epi_features=4, with_y=True)
    x, y = input_data.get_dataset()
    totaldata2[i] = np.expand_dims(x, axis=2)
    
dataArr2 = np.concatenate((totaldata2))
print(dataArr2.shape)
np.save("autoencoder_inputs_reg.npy",dataArr2)

datainverse2 = dataArr2[...,::-1]
print(dataArr2.shape)
#np.save("sav1_reg.npy",dataArr2)
print(dataArr2[0])

print(datainverse2.shape)
#np.save("sav2_reg.npy",datainverse2)
print(datainverse2[0])

Data2 = np.concatenate((datainverse2,dataArr2))
print(Data2.shape)
np.save("DATA_Regression.npy",Data2)

