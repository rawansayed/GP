import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.disable_v2_behavior()
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d
from tflearn.activations import relu , sigmoid, softmax
from tflearn.layers.normalization import batch_normalization 
from tflearn.data_utils import load_csv
from tflearn.initializations import uniform
from sklearn.model_selection import train_test_split
from tflearn.layers.estimator import regression
import pandas as pd 
import numpy as np
batch_size=256

from autoEncoder import create_auto_encoder

def get_all_tensor_names(filename):
    l=[i.name for i in tf.get_default_graph().get_operations()]
    # l = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    a_file = open(filename, "w")
    np.savetxt(a_file, l, delimiter=',', fmt='%s')
    a_file.close()

#######################################################################################################

tf.reset_default_graph()
auto_encoder = create_auto_encoder()
auto_encoder.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl",weights_only=True)

tf222= tf.get_default_graph()
# df=pd.DataFrame(X[0].reshape(8, 23))
# df.to_csv("./TrainingOutputs/inputs.csv")

# encode_decode = model.predict(X[0].reshape(1,8, 1, 23))
# output=np.array(encode_decode)
# output[:] = output[:]>0.5
# df=pd.DataFrame(output.reshape(8, 23))
# df.to_csv("./TrainingOutputs/outputs.csv")

# # get_all_tensor_names in side the model
# get_all_tensor_names("./testFiles/testEncoder.txt")


# ##### to get cerain tensor using name
encoder_channel_size = [23, 32, 64, 64, 256, 256]

# array_of_tensors=[]
# array_of_weights=[]
for layerNum in range(len(encoder_channel_size)):
    variables_to_Be_restored=[
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
        f"convEncoder_{layerNum}/W:0",
        # f"convEncoder_{layerNum}/W/Assign:0",
        f"convEncoder_{layerNum}/W/read:0",
        f"convEncoder_{layerNum}/Conv2D:0",
        f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
        f"BatchNormalizeEncoder_{layerNum}/beta:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0"
    ]
    if layerNum==0:
        variables_to_Be_restored=[
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
            f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
            f"convEncoder_{layerNum}/W:0",
            # f"convEncoder_{layerNum}/W/Assign:0",
            f"convEncoder_{layerNum}/W/read:0",
            f"convEncoder_{layerNum}/Conv2D:0",
            f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
            f"BatchNormalizeEncoder_{layerNum}/beta:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
            f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
            f"BatchNormalizeEncoder_{layerNum}/is_training:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
            f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
            f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
            f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
            f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
            f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
            f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
            f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0"
        ]
#     for variable in variables_to_Be_restored:
#         # print(layerNum,variable,len(variables_to_Be_restored))
#         try:
#             array_of_tensors.append(
#                 tf.Variable(
#                     tf.get_default_graph().get_tensor_by_name(
#                         variable)))
#         except:
#             array_of_weights.append(
#                 tf.Variable(
#                     tf.get_default_graph().get_tensor_by_name(
#                         variable).W))

#     for variable in variables_to_Be_restored:
#         i=0
#         j=0
#         print(layerNum,variable,len(variables_to_Be_restored))
#         try:
#             auto_encoder.set_weights(
#                 tf.get_default_graph().get_tensor_by_name(variable)
#                 , array_of_tensors[i] )
#             i+=1
        # except:
        #     auto_encoder.set_weights(
        #         tf.get_default_graph().get_tensor_by_name(variable)
#                 , array_of_weights[j])
#             j+=1
            
        # auto_encoder.set_weights(
        #     tf.get_default_graph().get_tensor_by_name(variable)
        #     , array_of_weights_in_all_layers[
        #         (layerNum+1)*variables_to_Be_restored.index(variable)])



##### To debug tensors
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl", tensor_name=None, all_tensors=True)


##### to get cerain tensor using name
encoder_channel_size = [23, 32, 64, 64, 256, 256]

array_of_tensors=[]
array_of_weights=[]
for layerNum in range(len(encoder_channel_size)):
    variables_to_Be_restored=[
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
        # f"convEncoder_{layerNum}/W:0",
        # f"convEncoder_{layerNum}/W/Assign:0",
        # f"convEncoder_{layerNum}/W/read:0",
        f"convEncoder_{layerNum}/Conv2D:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0"
    ]
    if layerNum==0:
        variables_to_Be_restored=[
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
            # f"convEncoder_{layerNum}/W:0",
            # f"convEncoder_{layerNum}/W/Assign:0",
            # f"convEncoder_{layerNum}/W/read:0",
            f"convEncoder_{layerNum}/Conv2D:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0",
            # "Sigmoid:0"
        ]
    for variable in variables_to_Be_restored:
        print(layerNum,variable,len(variables_to_Be_restored))
        # array_of_weights.append(
        #         tf.Variable(
        #             tf.get_default_graph().get_tensor_by_name(
        #                 variable).W))
        array_of_weights.append(
            # tf.constant_initializer(
                # np.array(
                    tf.Variable(
                        tflearn.variables.get_layer_variables_by_name(
                            f'convEncoder_{layerNum}')
                            )#[0]
                        )
                    # )
                # )
        # tf.constant_initializer

        # array_of_weights.append(
        #            tf.constant_initializer(tf.get_default_graph().get_tensor_by_name(
        #                  variable).W))   
print(array_of_weights[0])

for layerNum in range(len(encoder_channel_size)):
    variables_to_Be_restored=[
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
        # f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
        # f"convEncoder_{layerNum}/W:0",
        # f"convEncoder_{layerNum}/W/Assign:0",
        # f"convEncoder_{layerNum}/W/read:0",
        f"convEncoder_{layerNum}/Conv2D:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0"
    ]
    if layerNum==0:
        variables_to_Be_restored=[
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
            # f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
            # f"convEncoder_{layerNum}/W:0",
            # f"convEncoder_{layerNum}/W/Assign:0",
            # f"convEncoder_{layerNum}/W/read:0",
            f"convEncoder_{layerNum}/Conv2D:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
            # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
            # f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
            # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0",
            # "Sigmoid:0"
        ]
    # for variable in variables_to_Be_restored:
    #     print(layerNum,variable,len(variables_to_Be_restored))
    #     j=0
    #     a=tf.get_default_graph().get_tensor_by_name(variable)
        
    #     auto_encoder.trainer.session.run(
    #         a.assign(array_of_weights[j])
    #     )
    #     # auto_encoder.set_weights(a, array_of_weights[j])
    #     j+=1



# ########################################################################################################################
tf.reset_default_graph()
X=np.load("inputs.npy")
y=np.load("labels.npy")
# print(X.shape)
# print(y.shape)
# print(X[0],len(X),len(X[0]),len(X[0][0]))
X=X.reshape([-1,8, 1, 23])
y=y.reshape([-1,1, 1, 1])

# print(X[0],len(X),len(X[0]),len(X[0][0]))
# for i in range(36*6):
#     print(array_of_weights_in_all_layers[i])
X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X.shape)
# print(X_test.shape)
Y=Y.reshape([-1,])
# print(Y.shape)
Y_test=Y_test.reshape([-1,]) 
# print(Y_test.shape)
# print(len(X),len(X[0]),len(X[0][0]),X.shape)
# print(len(X_test),len(X_test[0]),len(X_test[0][0]),X_test.shape)
# print(len(Y),len(Y[0]),len(Y[0][0]),Y.shape)
# print(len(Y_test),len(Y_test[0]),len(Y_test[0][0]),Y_test.shape)




# Building the network
channel_size=[23, 32, 64, 64, 256, 256,512, 512, 1024, 2]
# betas=[uniform (shape=[1,channel_size[i]]) for i in range(len(channel_size))]
betas =  [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(len(channel_size))]
# print(betas[0])
AE = input_data(shape=[None,8, 1, 23], name='input')

encoder_channel_size = [23, 32, 64, 64, 256, 256]
for i in range(len(encoder_channel_size)):
    # array_of_weights[i] = tf.get_variable('get_variable'
    # , dtype=tf.float32, initializer=array_of_weights[i])
    # print(array_of_weights[i].value())
    # AE,encoder_channel_size[i], [1, 3],bias=False,activation=None,name=f"convEncoder_{i}"
    AE = conv_2d(AE,encoder_channel_size, [1, 3],bias=False,activation=None,name=f"convEncoder_{i}",weights_init=array_of_weights[i])
    AE = batch_normalization(AE,decay=0,name=f"BatchNormalizeEncoder_{i}",trainable=False)
    AE=sigmoid(AE)
    # AE = AE + betas[i]
    # AE = relu(AE)



cls_channel_size = [512, 512, 1024,2]
for i in range(len(cls_channel_size)-1):
    if i==0:
        
        AE = conv_2d(AE,cls_channel_size[i], [1, 3], strides=2,bias=False,activation=None,name=f"convCls_{i}",restore=False)
    if i==1:
        
        AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)
    if i==2:
        
        AE = conv_2d(AE,cls_channel_size[i], [1, 3],bias=False,activation=None,name=f"convCls_{i}",restore=False)#,padding='VALID'
    AE = batch_normalization(AE,decay=0.99,name=f"BatchNormalizeCls_{i}",restore=False)
    # AE = AE + betas[i]
    # AE = relu(AE)
    AE=sigmoid(AE)

# AE=AE.reshape([None,1,1,2])
AE = conv_2d(AE,cls_channel_size[3], [1, 1],bias=False,activation=None,name="convCls_3",restore=False)
# AE = AE + betas[i]
AE = tf.nn.softmax(AE)

# AE.set_shape([None,1,1,2])
print(AE.shape)
AE=tf.reshape(AE, (-1,1,1,8))
print(AE.shape)

AE = tf.squeeze(AE, axis=[1, 2])[:, 1]

# print(AE.shape)
# AE = tf.squeeze(AE, axis=[1, 2])[:, 1]

AE = regression(AE, optimizer='adam', learning_rate=0.0001
    , loss='categorical_crossentropy', name='target', restore=False)


# Training the network
model = tflearn.DNN(AE,tensorboard_verbose=0,
tensorboard_dir = './TrainingOutputs/classification/AE',
checkpoint_path = './TrainingOutputs/classification/AE/checkpoint')

get_all_tensor_names("./testFiles/testCls.txt")


# dynamically assign weights to a layer
# initialize layer weights from a numpy array
# how to write a layer initializer function in tensorflow
# what is a feed_dict in tensorflow

# Process:
# extract weights from layer as numpy array
# Either initialize or dynamically assign weights to new layers

model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl",weights_only=True)

encoder_channel_size = [23, 32, 64, 64, 256, 256]
for layerNum in range(len(encoder_channel_size)):
    variables_to_Be_restored=[
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/shape:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/min:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/max:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/RandomUniform:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/sub:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform/mul:0",
        f"convEncoder_{layerNum}/W/Initializer/random_uniform:0",
        f"convEncoder_{layerNum}/W:0",
        # f"convEncoder_{layerNum}/W/Assign:0",
        f"convEncoder_{layerNum}/W/read:0",
        # f"convEncoder_{layerNum}/Conv2D:0",
        f"BatchNormalizeEncoder_{layerNum}/beta/Initializer/Const:0",
        f"BatchNormalizeEncoder_{layerNum}/beta:0",
        # f"BatchNormalizeEncoder_{layerNum}/beta/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/beta/read:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/shape:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mean:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/stddev:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/RandomStandardNormal:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal/mul:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/Initializer/random_normal:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma:0",
        # f"BatchNormalizeEncoder_{layerNum}/gamma/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/gamma/read:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean/Initializer/zeros:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_mean/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_mean/read:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance/Initializer/Const:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/moving_variance/Assign:0",
        f"BatchNormalizeEncoder_{layerNum}/moving_variance/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Initializer/Const:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/is_training/read:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1/value:0",
        # f"BatchNormalizeEncoder_{layerNum}/Assign_1:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_t:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/switch_f:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/pred_id:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/mean/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/StopGradient:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/SquaredDifference:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance/reduction_indices:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/variance:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/moments/Squeeze_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/decay:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/sub/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/AssignMovingAvg_1/Switch:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Identity_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_1:0",
        # f"BatchNormalizeEncoder_{layerNum}/cond/Switch_2:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Merge:0",
        f"BatchNormalizeEncoder_{layerNum}/cond/Merge_1:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/add/y:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/add:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/Rsqrt:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_1:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/mul_2:0",
        f"BatchNormalizeEncoder_{layerNum}/batchnorm/sub:0",
        # f"BatchNormalizeEncoder_{layerNum}/batchnorm/add_1:0"
    ]
#     for variable in variables_to_Be_restored:
#         model.set_weights(
#             tf.get_default_graph().get_tensor_by_name(variable)
#             , array_of_weights_in_all_layers[
#                 (layerNum+1)*variables_to_Be_restored.index(variable)])

# for variable in variables_to_Be_restored:
#     j=0
#     tf.get_default_graph().get_tensor_by_name(
#         variable).assign(
#             tf.variable(
#                 array_of_weights[j]))
#     j+=1



model.fit({'input': X}, {'target': Y}, n_epoch=25,batch_size=batch_size,
validation_set=({'input': X_test}, {'target': Y_test}),
snapshot_step=1000,show_metric=True, run_id='convnet_mnist')

model.save("./TrainingOutputs/classification/clsModel/ClassificationModel.tfl")


