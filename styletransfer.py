# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:17:57 2018
Neural Style transfer code for DeepDream based style transfer
Note that in this exercise, we use L2 regularization instead of weight regularization
We define loss not in matching to a label, but by three subcomponents.

Before, loss was defined as things such as categorical_crossentropy etc. in the compile function,
Now we define the loss itself

To better understand parameters and inner workings
https://towardsdatascience.com/experiments-on-different-loss-configurations-for-style-transfer-7e3147eda55e
@author: Omistaja
"""

from keras.preprocessing.image import load_img, img_to_array

"""
Path to image youre using as the content, and also reference image
"""
target_image_path = 'c:/tensorflow_work/styletransfer/ghostref.jpg'
style_reference_image_path = 'c:/tensorflow_work/styletransfer/fear1.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

"""
Auxiliary functions for loading images into tensors and vice versa
"""

import numpy as np
from keras.applications import vgg19
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
"""
Zero-centering by removing the mean pixel value
from ImageNet. This reverses a transformation
done by vgg19.preprocess_input.
Converts images from 'BGR' to 'RGB'.
This is also part of the reversal of
vgg19.preprocess_input
"""
def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


"""
Load the model & apply to three images
"""
from keras import backend as K
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Pre trained VGG19 Model loaded.')

"""
Content loss here is decribed as the difference between generated vs original level
We will use this in gradient ascent to properly  backpropagated the computed final gradient for the generated image
To compute the content loss, you use only one upper layer—the block5_conv2 layer
"""
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

"""
Style loss contains the gram matrix defined here. more tba

style loss, you use a list of layers than spans both low-level and high-level layers. You
add the total variation loss at the end.
"""

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

"""
Variation loss tries to ensure consistency and spacial continuity, minimizing the pixelation
"""
def total_variation_loss(x):
    a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, 1:, :img_width - 1, :])
    b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

"""
 Make a dictioary for layers
"""

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

"""
These weights should be played around with to find best favourite output
"""
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.0003

"""
Now to combine it into a total weighted loss.
We start with the original loss, and then we add in more details
"""

loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(target_image_features,
                                      combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
    
loss += total_variation_weight * total_variation_loss(combination_image)

"""
Now we actually setup the gradient descent process to create a combination image thats ideal
grads here returns the loss with respect to the combination_image.
Fetch loss and grads is quite important, it is a function that it takes the combination_image tensor and returns loss and gradients with respect t
This is called by our evaluator class and then the loss and grads are extracted
"""
"""
Grads here is very important, it returns the change in loss with respect to image change
Loss has been previously defined already by comparing the two images
In order for it to reurn something however, we need to actually run the code, so that it can
Compare the product made to the target image. Running grads by itself doesnt gives shit
"""
grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

"""
Technically, you could calculate the loss and grads separately, which we'vedone bfore
But to speed things out, we do it in one class call

Create a class that wraps fetch_loss_and_grads
in a way that lets you retrieve the losses and
gradients via two separate method calls,
"""
class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
    
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

"""
Time to use gradient descent
"""

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
result_prefix = 'my_result'
iterations = 20
x = preprocess_image(target_image_path)
x = x.flatten()
""" Gradient descent will be revealed on Nov 5th"""
