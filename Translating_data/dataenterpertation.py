from ontarget import Episgt
import numpy as np


file_path = './database/paper_data-classification/paper_data/ontar/hct116_hart.episgt'
# file_path = './examples/eg_cls_on_target.episgt'
input_data = Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
X = np.expand_dims(x, axis=2)
# X = X.transpose([0, 2, 3, 1])
np.save("inputs.npy",X)
np.save("labels.npy",y)