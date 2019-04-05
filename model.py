from __future__ import print_function
import os
import keras
import time
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def zscore(X_train):
        #in case someone wants to normalize with z-score instead
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std)
        return X_train

NAME = "RapVsCountryWithTest0001DEF100epochs{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

checkpoint = ModelCheckpoint("./weights/weightsRapVsCountry.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')


train_datagen = image.ImageDataGenerator(
        rescale = 1./255,
        validation_split=0.2
#        shear_range = 0.2,
#        zoom_range= 0.4
)

train_generator = train_datagen.flow_from_directory(
        "./training/",
#        color_mode = "grayscale",
        target_size = (100, 100),
        batch_size = 32,
        shuffle = True,
        class_mode = "binary",
        subset='training')

val_generator = train_datagen.flow_from_directory(
        "./training/",
#        color_mode = "grayscale",
        target_size = (100, 100),
        class_mode = "binary",
        subset='validation')


model = Sequential([
Conv2D(64, (3, 3), input_shape=(100,100,3), padding="same", activation="relu"),
Conv2D(64, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(128, (3, 3), activation="relu", padding="same"),
Conv2D(128, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
Conv2D(256, (3, 3), activation="relu", padding="same"),
Conv2D(256, (3, 3), activation="relu", padding="same"),
Conv2D(256, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
Conv2D(512, (3, 3), activation="relu", padding="same"),
Conv2D(512, (3, 3), activation="relu", padding="same"),
Conv2D(512, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
Conv2D(512, (3, 3), activation="relu", padding="same"),
Conv2D(512, (3, 3), activation="relu", padding="same"),
Conv2D(512, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Flatten(),
Dense(4096, activation="relu"),
#                Dropout(0.2),
Dense(4096, activation="relu"),
#                Dropout(0.2),
Dense(1, activation="sigmoid")
])

print(model.summary())

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples // 32,
                              epochs=35,
                              verbose=1,
                              validation_data=val_generator,
                              validation_steps = val_generator.samples // 32,
                              callbacks=[tensorboard, checkpoint])

print(history.history.keys())
print(history.history.values())



x_train = []

for x in os.listdir("test")[6600:]:
        x_train.append(np.array(Image.open(x).convert("RGB")))
