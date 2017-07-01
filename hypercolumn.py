from matplotlib import pyplot as plt

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import theano
import cv2
import numpy as np
import scipy as sp

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD

from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
    
model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

im_original = cv2.resize(cv2.imread('abder.png'), (224, 224))
im = im_original.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
plt.imshow(im_converted)


#out = model.predict(im)
#plt.plot(out.ravel())

get_feature = theano.function([model.layers[0].input], model.layers[3].output(train=False), allow_input_downcast=False)
feat = get_feature(im)
plt.imshow(feat[0][2])

def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output(train=False) for li in layer_indexes]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)
    feature_maps = get_feature(instance)
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)

    return np.asarray(hypercolumns)

layers_extract = [3, 8]
hc = extract_hypercolumn(model, layers_extract, im)

ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)

m = hc.transpose(1,2,0).reshape(50176, -1)
kmeans = cluster.KMeans(n_clusters=2, max_iter=300, n_jobs=5, precompute_distances=True)
cluster_labels = kmeans .fit_predict(m)

imcluster = np.zeros((224,224))
imcluster = imcluster.reshape((224*224,))
imcluster = cluster_labels

plt.imshow(imcluster.reshape(224, 224), cmap="hot")
plt.show()
