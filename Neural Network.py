# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:00:26 2017

@author: XMKZ
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import pandas as pd

xdata = pd.read_csv('D:/大四下/data science/internship/house/X.csv', header=0)
ydata = pd.read_csv('D:/大四下/data science/internship/house/Y.csv', header=0)
ydata = np.array(ydata)
xdata = np.array(xdata)
X, y = xdata, np.log(ydata[:,4])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.fit_transform(X_test)

#X_train_std= X_train_std.astype('float32')
#X_test_std= X_test_std.astype('float32')
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')
y_train=y_train.astype('float32')
y_test= y_test.astype('float32')

'''
ydata = np.array(ydata)
xdata = np.array(xdata)
ydata = ydata-1
xdata= xdata.astype('float32')
ydata= ydata.astype('float32')
X_train = xdata[0:7600,:]
X_test = xdata[7601:10671,:]
y_train = ydata[0:7600]
y_test = ydata[7601:10671]
'''

model = Sequential([
    Dense(5, input_dim=13),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])

rmsprop = RMSprop(lr=0.0001)

model.compile(optimizer=rmsprop,
              loss='mae'
              )

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1000000, verbose=2, batch_size=10000)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test_std, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
'''
pre = model.predict_classes(X_test)
print(sum(pre==y_test)/3000)
'''