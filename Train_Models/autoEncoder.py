from sklearn.model_selection import train_test_split
from Build_Models.auto_encoder import create_auto_encoder
import numpy as np
import pandas as pd

batch_size=256
def main():
    X=np.load("inputs.npy")
    # X=X.reshape([-1,8, 1, 23])
    X = X.transpose([0, 2, 3, 1])

    # print(X[0])
    # print(X.shape)
    X, X_test, Y, Y_test = train_test_split(X, X, test_size=0.33, random_state=42)
    # print(X.shape)
    # print(X_test.shape)
    # print(Y.shape)
    # print(Y_test.shape)

    model =create_auto_encoder()

    model.fit({'input': X}, {'target': Y}, n_epoch=36,batch_size=batch_size,
    validation_set=({'input': X_test}, {'target': Y_test}),
    snapshot_step=1000,show_metric=True)

    
    model.save("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")
    
    # model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")

    df=pd.DataFrame(X[0].reshape(23, 8))
    df.to_csv("./TrainingOutputs/inputs.csv")

    encode_decode = model.predict(X[0].reshape(1,1, 23, 8))
    output=np.array(encode_decode)
    output[:] = output[:]>0.5
    df=pd.DataFrame(output.reshape(23, 8))
    df.to_csv("./TrainingOutputs/outputs.csv")

main()