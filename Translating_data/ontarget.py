import pandas as pd
import numpy as np
from operator import add
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

__all__ = ['Sgt', 'Episgt', 'Epiotrt']
# sgt:ontarget_sequence only, episgt:ontarget, epiotrt:offtarget

ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
        }

epimap = {'A': 1, 'N': 0}
#represent presence/absence of 4 other channels "epigentic features:dnase,ctcf,rrbs,h3k4me3"

def get_seqcode(seq):  # [1 0 0 0], [0 1 0 0] ..... etc
    return np.array(reduce(add, map(lambda c: ntmap[c], seq.upper()))).reshape(
        (1, len(seq), -1))

def get_epicode(eseq):
    return np.array(list(map(lambda c: epimap[c], eseq))).reshape(1, len(eseq), -1)

# x= get_seqcode('ACGTTAGCAGTTTGATGGCATGG')
# print (x)
# y= get_epicode('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANNNNNNNNNNNNNNNNNNNNNNN')
# print (y)
class Sgt:
    def __init__(self, fpath, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._with_y = with_y
        self._num_cols = 2 if with_y else 1
        print("self._num_cols",self._num_cols)
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        print("self._cols",self._cols)
        self._df = self._ori_df[self._cols]
        print(self._df)

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        print("X_seq",x_seq)
        x = x_seq.astype(dtype=x_dtype)
        x = x.transpose(0, 2, 1)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            print("Y ",y)
            return x, y
        else:
            return x

class Episgt:
    def __init__(self, fpath, num_epi_features, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._num_epi_features = num_epi_features
        self._with_y = with_y
        self._num_cols = num_epi_features + 2 if with_y else num_epi_features + 1
        print("self._num_cols",self._num_cols)
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        print("self._ori_df.columns",self._ori_df.columns)
        print("self._cols",self._cols)
        self._df = self._ori_df[self._cols]
        print("Length self._df",len(self._df))
        print(self._df)

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        x_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        print("X_seq", x_seq)
        print("X_epis", x_epis)
        x = np.concatenate([x_seq, x_epis], axis=-1).astype(x_dtype)
        print("X", x)
        x = x.transpose(0, 2, 1)
        print("X_transpose", x)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            print("Y",y)
            return x, y
        else:
            return x

# file_path = 'eg_cls_on_target.episgt'
# file_path = 'eg_reg_on_target.repisgt'
# input_data = Episgt(file_path, num_epi_features=4, with_y=True)
# x, y = input_data.get_dataset()
# x = np.expand_dims(x, axis=2)
# x_train, x_test, y_train, y_test = train_test_split(x, y) #train75%,testing25%
# optimized_data=[x_train, x_test, y_train, y_test]
# np.save('./eg_1_reg_on_target.repisgt', optimized_data, allow_pickle=True)

# x_train = np.reshape(x_train, (-1, 100))
# print("X_train_reshape IS ----------",len(x_train))
# x_test = np.reshape(x_test, (-1, 100))
# print("X_test_reshape IS ----------",len(x_test))
# batch_size=50
# Reserve 10,000 samples for validation.
# x_val = x_train[-80:]
# y_val = y_train[-80:]
# x_train = x_train[:-25]
# y_train = y_train[:-25]
# print("X_train IS ----------",x_train)
# print("Y_train IS ----------",y_train)
# print("X_val IS ----------",x_val)
# print("Y_val IS ----------",y_val)
# print("Length X_VAL",len(x_val))

# # Prepare the training dataset.
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# # Prepare the validation dataset.
# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# print("VAL_DATA",val_dataset)
# val_dataset = val_dataset.batch(batch_size)
# print("VAL_DATA",val_dataset)



# x_train, x_test, y_train, y_test = train_test_split(x, y) #train75%,testing25%
# optimized_data=[x_train, x_test, y_train, y_test]