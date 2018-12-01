from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np

# get the data
filename = 'test_data.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def getData(filename):
    # images are 48x48
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            X.append([int(p) for p in row[0].split()])
    X = np.array(X) / 255.0
    return X


X = getData(filename)


# Keras with Tensorflow backend
N, D = X.shape
X = X.reshape(N, 48, 48, 1)

print(X.shape)

def baseline_model_saved():
    # load json and create model
    json_file = open('model_4layer_2_2_pool.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights from h5 file
    model.load_weights("model_4layer_2_2_pool.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


print("Load model from disk")
model = baseline_model_saved()

# Model will predict the probability values for 7 labels for a test image
score = model.predict(X)
print(model.summary())

new_X = [np.argmax(item) for item in score]
print("The predicted values are {}".format(new_X))

