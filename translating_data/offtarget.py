import pandas as pd
import numpy as np
from operator import add
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
# import tensorflow as tf

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

class Epiotrt:
    def __init__(self, fpath, num_epi_features, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._num_epi_features = num_epi_features
        self._with_y = with_y
        self._num_cols = num_epi_features * 2 + 3 if with_y else num_epi_features * 2 + 2
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
        x_on_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        # print("x_on_seq",x_on_seq)
        x_on_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        # print("x_on_epis",x_on_epis)
        x_on = np.concatenate([x_on_seq, x_on_epis], axis=-1).astype(x_dtype)
        # print("X_on", x_on)
        x_on = x_on.transpose(0, 2, 1)
        # print("X_on Transpose", x_on)
        x_off_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[1 + self._num_epi_features]])))
        # print("x_off_seq",x_off_seq)
        x_off_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[2 + self._num_epi_features: 2 + self._num_epi_features * 2]], axis=-1)
        # print("x_off_epis",x_off_epis)
        x_off = np.concatenate([x_off_seq, x_off_seq], axis=-1).astype(x_dtype)
        # print("X_off", x_off)
        x_off = x_off.transpose(0, 2, 1)
        # print("X_off Transpose", x_off)


        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            return (x_on, x_off), y
        else:
            return (x_on, x_off)


# file_path = 'eg_cls_off_target.epiotrt'

# file_path = 'eg_reg_off_target.repiotrt'
# input_data = Epiotrt(file_path, num_epi_features=4, with_y=True)
# (x_on, x_off), y = input_data.get_dataset()
# x_on = np.expand_dims(x_on, axis=2)  # shape(x) = [100, 8, 1, 23]
# x_off = np.expand_dims(x_off, axis=2)  # shape(x) = [100, 8, 1, 23]
# x_train_on, x_test_on, y_train, y_test = train_test_split(x_on, y) #train75%,testing25%
# x_train_off, x_test_off, y_train, y_test = train_test_split(x_off, y)
# optimized_data=[x_train_on,x_train_off, x_test_on,x_test_off, y_train, y_test]
# np.save('./eg_1_reg_off_target.repiotrt', optimized_data, allow_pickle=True)

# print("X_ON IS ----------",x_on)
# print("X_OFF IS ----------",x_off)
# print("Y IS ----------",y)