import pandas as pd
from google.colab import drive
drive.mount('/content/drive/')

!ls

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

batch_size = 200
num_classes = 10
epochs = 10

img_rows, img_cols = 128, 128

# the data, split between train and test sets
y_train = pd.read_csv('/content/drive/My Drive/ML_Project3/train_max_y.csv').Label.values
x_train = pd.read_pickle('/content/drive/My Drive/ML_Project3/train_max_x')
x_test = pd.read_pickle('/content/drive/My Drive/ML_Project3/test_max_x')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

def binarize(images, threshold=220, invert=True):
    upper = 1
    lower = 0
    if invert:
        upper, lower = lower, upper
    return np.where(images > threshold, upper, lower)

x_train = binarize(x_train)
x_test = binarize(x_test)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Plot the images
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', input_shape=input_shape))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

model.evaluate(x_train,y_train)

model.save('/content/drive/My Drive/ML_Project3/my_model3.h5')
model.save('models/bad_model_15_epochs.h5')

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
