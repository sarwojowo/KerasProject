#python resnetcustom.py --file images/office.png
from __future__ import print_function
import numpy as np
import json
import os
import time

from keras import backend as K
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.utils.data_utils import get_file



CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://s3.amazonaws.com/deep-learning-models/'
                    'image-models/imagenet_class_index.json')


def preprocess_input(x, dim_ordering='default'): 
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):

    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results


def is_valid_file(parser, arg):

    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        type=lambda x: is_valid_file(parser, x),
                        help="Classify image",
                        metavar="IMAGE",
                        required=True)    
    return parser



if __name__ == "__main__":
    args = get_parser().parse_args()

    # Load model
    model = ResNet50(include_top=True, weights='imagenet')

    img_path = args.filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
   # print('Input image shape:', x.shape)
    t0 = time.time()
    preds = model.predict(x)
    t1 = time.time()
    print("Prediction time: {:0.3f}s".format(t1 - t0))
    for wordnet_id, class_name, prob in decode_predictions(preds)[0]:
        print("{wid}\t{prob:>6}%\t{name}".format(wid=wordnet_id,
                                                 name=class_name,
                                                 prob="%0.2f" % (prob * 100)))
