import numpy as np
import scipy.misc


from keras.applications import VGG16
from keras.applications import VGG19

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def vgg16(img_path):
    print('Loading VGG16')
    model = VGG16(weights='imagenet', include_top=False )
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    print(features)
    return features
    
def resizing_image(img):
    print('resizing an input image')
    imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img), (150, 150)),
                         (2,0,1)).astype('float32')]
    print(type(imgs))
    print(imgs[0].shape)
    return np.array(imgs)/255 

def load_model(model_def_fname, model_weight_fname):
    model = model_from_json(open(model_def_fname).read())
    model.load_weights(model_weight_fname)
    return model
 
def head_model(bottom_features):
    print('loading the top model')
    model = load_model(classifier_path, weights_path)
    prediction = model.predict_classes(bottom_features)
    return prediction
       
if __name__ == '__main__':

    bottom_features = vgg16(image_path)
    
    head_prediction = head_model(bottom_features)
    
    print('prediction ', head_prediction)
    
    print('the program is done')