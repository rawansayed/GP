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
        # print("self._num_cols",self._num_cols)
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        # print("self._cols",self._cols)
        self._df = self._ori_df[self._cols]
        # print(self._df)

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        # print("X_seq",x_seq)
        x = x_seq.astype(dtype=x_dtype)
        x = x.transpose(0, 2, 1)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            # print("Y ",y)
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
        #print("self._num_cols",self._num_cols)
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        # print("self._ori_df.columns",self._ori_df.columns)
        # print("self._cols",self._cols)
        self._df = self._ori_df[self._cols]
        # print("Length self._df",len(self._df))
        # print(self._df)

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        x_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        # print("X_seq", x_seq)
        # print("X_epis", x_epis)
        x = np.concatenate([x_seq, x_epis], axis=-1).astype(x_dtype)
        # print("X", x)
        x = x.transpose(0, 2, 1)
        # print("X_transpose", x)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            # print("Y",y)
            return x, y
        else:
            return x

