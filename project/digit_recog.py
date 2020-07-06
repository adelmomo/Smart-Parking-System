import numpy as np
from skimage import io
from skimage import color
import cv2
np.random.seed(123)  # for reproducibility
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
import itertools
from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import os
from sklearn.metrics import roc_curve, auc,confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import tensorflow as tf

#from digit_recog_load import *
from tensorflow.keras.callbacks import ModelCheckpoint

n=100
m=100
c=1
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3,3),strides=2,input_shape=(n,m,c)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
checkpoint = ModelCheckpoint('demo.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.load_weights('digitrecog.h5')
#Y_pred=model.predict(X_test)
#ROC(Y_pred,Y_test,10)
#Confusion()
#ROC(Y_pred,Y_test,10)
#history=model.fit(trainx,trainy,batch_size=200,epochs=100, callbacks=callbacks_list,validation_data=(testx,testy),verbose=2)
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=50),
                     #epochs=50,verbose=2,validation_data=(X_test,Y_test))


##model = Sequential()
##model.add(Conv2D(32, kernel_size=(3,3),
##                 strides=2,
##                 activation='relu',
##               input_shape=(n,m,c)))
##model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu'))
##model.add(Dropout(0.25))
##model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu'))
##model.add(Dropout(0.25))
##model.add(Flatten())
##model.add(Dense(1024, activation='relu'))
##model.add(Dropout(0.25))
##model.add(Dense(128,activation='relu'))
##model.add(Dropout(0.25))
##model.add(Dense(10, activation='softmax'))
##model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
##model.fit(X_train,Y_train,batch_size=30,epochs=50,validation_data=(X_test,Y_test),verbose=2)
##
