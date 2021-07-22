import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras import initializers
import tensorflow.keras.backend as K


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.
    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.
    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v

class AttentionAugmentation2D(Layer):

    def __init__(self, depth_k, depth_v, num_heads, relative=True, **kwargs):
        """
        Applies attention augmentation on a convolutional layer
        output.
        Args:
            depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative: bool, whether to use relative encodings.
        Raises:
            ValueError: if depth_v or depth_k is not divisible by
                num_heads.
        Returns:
            Output tensor of shape
            -   [Batch, Height, Width, Depth_V] if
                channels_last data format.
            -   [Batch, Depth_V, Height, Width] if
                channels_first data format.
        """
        super(AttentionAugmentation2D, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (
                depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                             'Given depth_k = %d, num_heads = %d' % (
                             depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                             'Given depth_v = %d, num_heads = %d' % (
                                 depth_v, num_heads))

        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v,
                                                           input_shape)

        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads

            if dk_per_head == 0:
                print('dk per head', dk_per_head)

            self.key_relative_w = self.add_weight('key_rel_w',
                                                  shape=[2 * width - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))

            self.key_relative_h = self.add_weight('key_rel_h',
                                                  shape=[2 * height - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))

        else:
            self.key_relative_w = None
            self.key_relative_h = None

    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5)

        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = K.reshape(q, K.stack(qk_shape))
        flat_k = K.reshape(k, K.stack(qk_shape))
        flat_v = K.reshape(v, K.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = K.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]
        attn_out_shape = K.stack(attn_out_shape)
        attn_out = K.reshape(attn_out, attn_out_shape)
        attn_out = self.combine_heads_2d(attn_out)
        # [batch, height, width, depth_v]

        if self.axis == 1:
            # return to [batch, depth_v, height, width] for channels first
            attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        attn_out.set_shape(self.compute_output_shape(self._shape))

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = K.shape(ip)

        # batch, height, width, channels for axis = -1
        tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = K.stack([batch, height, width,  self.num_heads, channels // self.num_heads])
        split = K.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)

        return split

    def relative_logits(self, q):
        shape = K.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            K.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = K.zeros(K.stack([B, Nh, L, 1]))
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros(K.stack([B, Nh, L - 1]))
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = K.stack(shape[:-2] + [a * b])
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def split_heads_2d(self, ip):
    tensor_shape = K.shape(ip)

    # batch, height, width, channels for axis = -1
    tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

    batch = tensor_shape[0]
    height = tensor_shape[1]
    width = tensor_shape[2]
    channels = tensor_shape[3]

    # Save the spatial tensor dimensions
    self._batch = batch
    self._height = height
    self._width = width

    ret_shape = K.stack([batch, height, width,  self.num_heads, channels // self.num_heads])
    split = K.reshape(ip, ret_shape)
    transpose_axes = (0, 3, 1, 2, 4)
    split = K.permute_dimensions(split, transpose_axes)

    return split







def attention(inputs, depth_k, depth_v, num_heads, relative=True):
    #init
    axis = -1













    #call
    if axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

        q, k, v = tf.split(inputs, [depth_k, depth_k, depth_v], axis=-1)

        q = split_heads_2d(q)
        k = split_heads_2d(k)
        v = split_heads_2d(v)

        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5)

        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = K.reshape(q, K.stack(qk_shape))
        flat_k = K.reshape(k, K.stack(qk_shape))
        flat_v = K.reshape(v, K.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = K.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]
        attn_out_shape = K.stack(attn_out_shape)
        attn_out = K.reshape(attn_out, attn_out_shape)
        attn_out = self.combine_heads_2d(attn_out)
        # [batch, height, width, depth_v]

        if self.axis == 1:
            # return to [batch, depth_v, height, width] for channels first
            attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        attn_out.set_shape(self.compute_output_shape(self._shape))

        return attn_out



# import tensorflow as tf
# from token import SLASH
# from typing import ClassVar
# from tflearn.metrics import Accuracy
# import tensorflow.compat.v1 as tf
# from tensorflow.keras.layers import Attention

# import tflearn
# from tflearn.layers.core import input_data , activation
# from tflearn.layers.conv import conv_2d, max_pool_1d
# from tflearn.activations import relu , sigmoid, softmax
# from tflearn.layers.normalization import batch_normalization 
# from tflearn.data_utils import load_csv
# from tflearn.initializations import uniform
# from tflearn.layers.estimator import regression
# from sklearn.model_selection import train_test_split
# from tflearn.layers.core import dropout
# from tflearn.metrics import accuracy
# from tflearn.layers.core import reshape
# import pandas as pd 
# import numpy as np
# from tflearn import variables as vs
# from tflearn.layers.recurrent import lstm
# from numpy.core.fromnumeric import reshape
# import tensorflow.compat.v1 as tf
# # tf.enable_eager_execution()
# import tflearn
# from sklearn.model_selection import train_test_split
# import pandas as pd 
# import numpy as np
# batch_size=256
# from sklearn.metrics import roc_auc_score












# B=256 #batch_size
# W=1 #second dimention
# H= 47879 #first dimention Training data size for encoder
# # H=  #first dimention Training data size for classification


# def split_heads_2d(inputs, Nh):
#     s = inputs.shape[:-1]

#     ret_shape = s + [Nh, s // Nh]
#     split = tf.reshape(inputs, ret_shape)
#     return tf.transpose(split, [0, 3, 1, 2, 4])


# def combine_heads_2d(inputs):
#     transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
#     a, b = transposed.shape[-2:]
#     ret_shape = transposed.shape[:-2] + [a * b]
#     return tf.reshape(transposed, ret_shape)


# def compute_flat_qkv(inputs, dk, dv, Nh):
#     N, H, W, _ = inputs.shape
#     qkv = tf.layers.conv2d(inputs, 2*dk + dv, 1)
#     q, k, v = tf.split(qkv, [dk, dk, dv], axis=3)
#     q = split_heads_2d(q, Nh)
#     k = split_heads_2d(k, Nh)
#     v = split_heads_2d(v, Nh)
#     dkh = dk // Nh
#     q *= dkh ** -0.5
#     flat_q = tf.reshape(q, [N, Nh, H * W, dk])
#     flat_k = tf.reshape(k, [N, Nh, H * W, dk])
#     flat_v = tf.reshape(v, [N, Nh, H * W, dv])
#     return flat_q, flat_k, flat_v

# def rel_to_abs(x):
#     B, Nh, L, = x.shape
#     col_pad = tf.zeros((B, Nh, L, 1))
#     x = tf.concat([x, col_pad], axis=3)
#     flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
#     flat_pad = tf.zeros((B, Nh, L-1))
#     flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
#     final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
#     final_x = final_x[:, :, :L, L-1:]
#     return final_x


# def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
#     rel_logits = tf.einsum('bhxyd,md−>bhxym', q, rel_k)
#     rel_logits = tf.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
#     rel_logits = rel_to_abs(rel_logits)
#     rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
#     rel_logits = tf.expand_dims(rel_logits, axis=3)
#     rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
#     rel_logits = tf.transpose(rel_logits, transpose_mask)
#     rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
#     return rel_logits


# def relative_logits(q):
#     dk = q.shape[-1]
#     key_rel_w = tf.get_variable(
#         'key_rel_w', shape=(2*W-1, dk),
#         initializer=tf.random_normal_initializer(dk**-0.5)
#     )
#     rel_logits_w = relative_logits_1d(
#         q, key_rel_w, H, W, [0, 1, 2, 4, 3, 5]
#     )
#     key_rel_h = tf.get_variable(
#         'key_rel_h', shape=(2*H-1, dk),
#         initializer=tf.random_normal_initializer(dk**-0.5)
#     )
#     rel_logits_h = relative_logits_1d(
#         tf.transpose(q, [0, 1, 3, 2, 4]),
#         key_rel_h, W, H, [0, 1, 4, 2, 5, 3]
#     )
#     return rel_logits_h, rel_logits_w




# def augmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
#     conv_out = tf.layers.conv2d(X, Fout - dv, k)
#     flat_q, flat_k, flat_v = compute_flat_qkv(X, dk, dv,Nh)
#     logits = tf.matmul(flat_q, flat_k, transpose_b=True)
#     if relative:
#         h_rel_logits, w_rel_logits = relative_logits(q)
#         logits += h_rel_logits
#         logits += w_rel_logits
#     weights = tf.nn.softmax(logits)
#     attn_out = tf.matmul(weights, flat_v)
#     attn_out = tf.reshape(flat_v, [B, Nh, H, W, dv // Nh])
#     attn_out = combine_heads_2d(flat_v)
#     attn_out = tf.layers.conv2d(attn_out, dv, 1)
#     return tf.concat([conv_out, attn_out], axis=3)


















# def create_classification_model():

#     CLS = input_data(shape=[None,1, 23, 8], name='input')
#     print(CLS.shape)
#     CLS =Attention(CLS)
#     print(CLS.shape)
#     CLS = tf.squeeze(CLS, axis=[1, 2])[:, 1]
#     print(CLS.shape)

#     # we define our optimizer and loss functions and learning rate in the regression layer 
#     CLS = regression(CLS, optimizer='adam', learning_rate=0.001,metric=accuracy()
#         , loss='binary_crossentropy', name='target', restore=False)
#     # binary_crossentropy
#     # categorical_crossentropy
#     # roc_auc_score


#     # creating the model
#     model = tflearn.DNN(CLS,tensorboard_verbose=0)

#     return model


# def main():
#     # loading data
#     x=np.load("inputs_cls.npy")
#     y=np.load("labels_cls.npy")
#     x = x.transpose([0, 2, 3, 1])
#     X, X_test, Y, Y_test = train_test_split(x, y, test_size=0.33, random_state=42,shuffle=True)
#     X_div, X_test, Y_div, Y_test = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

#     model=create_classification_model()

#     model.fit({'input': X}, {'target': Y}, n_epoch=200,batch_size=batch_size,
#     validation_set=({'input': X_div}, {'target': Y_div}),
#     snapshot_step=1000,show_metric=True)



# main()








