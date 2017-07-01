# Trained image classification models for Keras

This repository contains code for the following Keras models:

- VGG16Custom
- VGG19Custom
- ResNet50Custom


All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at `~/.keras/keras.json`. For instance, if you have set `image_dim_ordering=tf`, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='imagenet'` argument in model constructor for all image models, `weights='msd'` for the music tagging model). Weights are automatically downloaded if necessary, and cached locally in `~/.keras/models/`.

Please follow https://github.com/fchollet/keras for original project
