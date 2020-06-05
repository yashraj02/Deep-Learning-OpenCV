import numpy as np
from numpy import genfromtxt

data = genfromtxt('BankNote_Authentication.csv', delimiter=',')
X = data[1:, 0:4]
y = data[1:, 4]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaleObject = MinMaxScaler()
scaleObject.fit(x_train)
print(x_test)
scaled_xtrain = scaleObject.transform(x_train)
scaled_xtest = scaleObject.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #'accuracy' gives accuracy metrices ie acc & loss

model.fit(scaled_xtrain,y_train,epochs=5000,verbose=2) #verbose prints info for eeach epoch

from sklearn.metrics import confusion_matrix,classification_report
ypred = model.predict_classes(scaled_xtest)
confusion_matrix(y_test,ypred)
print(classification_report(y_test,ypred))
model.save('mymodel')
from keras.models import load_model
newmodel = load_model('mymodel')