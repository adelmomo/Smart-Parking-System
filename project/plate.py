import numpy as np
import cv2 
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc,confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import itertools
#from plate_load import *

n=100
m=100
c=1
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3,3),
                 strides=2,
               input_shape=(n,m,c)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(BatchNormalization())
model.add(Activation('softmax'))
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
checkpoint = ModelCheckpoint('plate123.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.load_weights('plate.h5')
# 9. Fit model on training data
#history=model.fit(X_train, Y_train,batch_size=100,callbacks=callbacks_list,epochs=300,validation_data=(X_test,Y_test),verbose=2)
#Confusion()

