# USAGE
# python classify_standart.py --image images/office.png --model vgg16
# thanks for Dr. adryan at http://www.pyimagesearch.com  my best teacher.

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import VGG16

from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import time

# vgg-16_custom
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# this model put from keras - aplication - model
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"resnet": ResNet50
}

if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
  
t0 = time.time()
preds = model.predict(image)
t1 = time.time()
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

print("Prediction time: {:0.3f}s".format(t1 - t0))

