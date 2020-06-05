# from keras.datasets import mnist
# (xtrain,ytrain), (xtest,ytest) = mnist.load_data()
print(xtrain.shape)
#scaling
scales_xtrain = xtrain/xtrain.max()
scaled_xtest = xtest/xtest.max()

from keras.utils.np_utils import to_categorical
y_cat_ytrain = to_categorical(ytrain)
y_cat_ytest = to_categorical(ytest)

