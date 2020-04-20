import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

#Load Data
data = pd.read_csv("/content/drive/My Drive/Road to AI Engineer/Tensorflow/fer2013.csv")
data.head()
data.info()
#Feature
num_features = 48
num_labels = 7
batch_size = 32
epochs = 10
width, height = 48, 48
#Preprocessing Data
def fer2013_to_X():
    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array
    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""

    X = []
    pixels_list = data["pixels"].values

    for pixels in pixels_list:
        single_image = np.reshape(pixels.split(" "), (width, height)).astype("float")
        X.append(single_image)

    # Convert list to 4D array:
    X = np.expand_dims(np.array(X), -1)

    # Normalize image data:
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    return X
#Get Features
X = fer2013_to_X()
X.shape
y = pd.get_dummies(data['emotion']).values
y.shape


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)


np.save('modXtest', X_test)
np.save('modytest', y_test)

#Build Model CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))\

model.summary()

#Compile model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

#Create Callback
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
filepath="weights-best-file.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose = 1, save_best_only = True,  mode = 'max')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

#Trainning model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks = [lr_reducer, tensorboard, early_stopper, checkpoint],
          validation_data=(X_valid, y_valid),
          shuffle=True)
#Get Accuracy and Loss
scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

#Saving the  model to  use it later on
fer_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model.h5")

