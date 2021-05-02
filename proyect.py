import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import utils as np_utils
from tensorflow.keras.datasets import cifar10

# Set random seed for purposes of reproducibility
seed = 21

# loading in the data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
# in order to normalize the data we can simply divide the image values by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# Creating model
model = Sequential()

# first convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))

# esto es nada más para prevenir overfitting
# 0.2 means it drops 20% of the existing connections
model.add(Dropout(0.2))

# Batch Normalization normalizes the inputs heading into the next layer, 
# ensuring that the network always creates activations with the 
# same distribution that we desire
model.add(BatchNormalization())

# another convolutional layer
# but the filter size increases so the network 
#  can learn more complex representations
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())


# Now --> Flatten then data
model.add(Flatten())
model.add(Dropout(0.2))


# First Densely connected layer (256 neurons)
model.add(Dense(256, kernel_constraint=MaxNorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=MaxNorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))


epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# np.random.seed(seed)
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# # Model evaluation
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
