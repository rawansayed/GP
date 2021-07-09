from sklearn.model_selection import train_test_split
from Build_Models.auto_encoder import create_auto_encoder
import numpy as np
import pandas as pd

batch_size=256
def main():
    # loading data
    X=np.load("DATA_Classification.npy")
    # X=X.reshape([-1,8, 1, 23])
    X = X.transpose([0, 2, 3, 1])
    # print(X[0])
    # print(X.shape)

    # Creating train and development data  // I did not make test data
    X, X_dev, Y, Y_dev = train_test_split(X, X, test_size=0.33, random_state=42)
    # print(X.shape)
    # print(X_dev.shape)
    # print(Y.shape)
    # print(Y_dev.shape)
    X = X 
    Y=X
    X_dev, X_test= train_test_split(X_dev, test_size=0.02, random_state=42)
    
    X_dev = X_dev 
    Y_dev = X_dev

    Y_test=X_dev
    
    # Creating model
    model =create_auto_encoder()

    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X}, {'target': Y}, n_epoch=25,batch_size=batch_size,
    validation_set=({'input': X_dev}, {'target': Y_dev}),
    snapshot_step=1000,show_metric=True)

    # save the model
    model.save("./TrainingOutputs/autoencoder/autoencoderModel/Classification_Model/model.tfl")
    
    # model.load("./TrainingOutputs/autoencoder/autoencoderModel/model.tfl")
    def accuracy(y_pred,Y_output):

        return


    df=pd.DataFrame(X_test[0].reshape(23, 8))
    df.to_csv("./TrainingOutputs/inputs_classification.csv")

    encode_decode = model.predict(X_test[0].reshape(1,1, 23, 8))
    output=np.array(encode_decode)
    output[:] = output[:]>0.5
    df=pd.DataFrame(output.reshape(23, 8))
    df.to_csv("./TrainingOutputs/outputs_classification.csv")
    # print(accuracy())
    # print(accuracy())
    # print(accuracy())

main()