from __future__ import print_function
import keras
import time
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

def zscore(X_train):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std)
        return X_train

NAME = "test{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import skimage

def my_gen(_dir, _actualdir):
        for _file in _dir:
                img = Image.open(_actualdir+_file)
                img = img.convert("L")
#                plt.imshow(img)
#                plt.show()

                arr = np.array(img)
#                print("ARRAY:", arr)
#                flat = arr.ravel()
                flat = arr.reshape(100,100,1)
#                print("FLAT : ",flat)
                yield (flat, _file)


vecs = []

print(os.getcwd())

for x in my_gen(os.listdir("/dir/to/group/a/"), "/dir/to/group/a/"):
    vecs.append(x)

vecs2 = []

for x in my_gen(os.listdir("/dir/to/group/b"), "/dir/to/group/b/"):
    vecs2.append(x)
#print(vecs)

vecs = [x[0] for x in vecs[:1800]]
vecs2 = [x[0] for x in vecs2[:1800]]
print(len(vecs2), len(vecs))
x_train = vecs+vecs2
x_train = np.array(x_train)
x_train = x_train.astype('float32')
x_train = zscore(x_train)
yecs = [1]*100
yecs1 = [0]*100


y_train = yecs+yecs1
y_train = np.array(y_train)
y_train = to_categorical(y_train, 2)
#y_train = to_categorical(y_train)

length = len(y_train)
mask = np.random.choice(length, length, replace=False)

x_train = x_train[mask]
y_train = y_train[mask]

print(len(x_train), len(y_train))



from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)[:1000]

x_train = x_train.astype('float32')[:1000]
x_test = x_test.astype('float32')
#x_train /= 255
x_test /= 255
x_train = normalize(x_train)



model = Sequential([
Conv2D(64, (3, 3), input_shape=x_train.shape[1:], padding="same", activation="relu"),
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
                Dropout(0.2),
Dense(4096, activation="relu"),
                Dropout(0.2),
Dense(10, activation="softmax")
])

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=2, shuffle=True,
                    verbose=1, validation_split=0.2,
                    callbacks=[tensorboard])

print(history.history.keys())
print(history.history.values())
print(model.summary())

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations1 = activation_model.predict(x_train[-30].reshape(1,32,32,3))
activations2 = activation_model.predict(x_train[30].reshape(1,32,32,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1



display_activation(activations1, 4,4,1)

plt.show()

display_activation(activations1, 4,4,2)

plt.show()

display_activation(activations2, 4,4,6)

plt.show()

display_activation(activations2, 4,4,1)

plt.show()
