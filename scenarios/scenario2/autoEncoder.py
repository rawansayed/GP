from sklearn.model_selection import train_test_split
from Build_Models.auto_encoder import create_auto_encoder
import numpy as np
import pandas as pd
# --------------- loading the location where we will save data into --------------- #
save_location=__file__[:__file__.rindex("/")]
print("save location:",save_location)

batch_size=256

def autoencoder():

    # loading data
    data=np.load(f"./training_data/DATA_AutoEncoder.npy")

    print("data auto encoder shape:",data.shape)
    # Reshape data
    data = data.transpose([0, 2, 3, 1])
    print("data auto encoder shape:",data.shape)


    X_train, X_dev = train_test_split(data, test_size=0.33, random_state=42)

    Y_train=X_train

    X_dev, X_test= train_test_split(X_dev, test_size=0.02, random_state=42)

    Y_dev = X_dev
    Y_test=X_test
    
    # Creating model
    model =create_auto_encoder()

    # start training with input as the X train data and target as Y train data
    # and validate/develop over X_dev and Y_dev
    model.fit({'input': X_train}, {'target': Y_train}, n_epoch=25,batch_size=batch_size,
    validation_set=({'input': X_dev}, {'target': Y_dev}),
    snapshot_step=1000,show_metric=True)

    # save the model
    model.save(f"{save_location}/autoencoderModel/model.tfl")

    df=pd.DataFrame(X_test[0].reshape(23, 8))
    df.to_csv(f"{save_location}/autoencoderstatistics/input_test_autoencoder.csv")

    encode_decode = model.predict(X_test[0].reshape(1,1, 23, 8))
    output=np.array(encode_decode)
    output[:] = output[:]>0.5
    df=pd.DataFrame(output.reshape(23, 8))
    df.to_csv(f"{save_location}/autoencoderstatistics/output_test_autoencoder.csv")


autoencoder()