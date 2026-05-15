#!/usr/bin/env python3
"""Module for Neural Style Transfer - Task 6: Content Cost"""
import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class.

        Args:
            style_image: numpy.ndarray - image used as style reference
            content_image: numpy.ndarray - image used as content reference
            alpha: float - weight for content cost
            beta: float - weight for style cost
        """
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) \
                or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image: numpy.ndarray of shape (h, w, 3) - image to be scaled

        Returns:
            tf.Tensor of shape (1, h_new, w_new, 3) - scaled image
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0), size=(h_new, w_new))
        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)

        return rescaled

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as a base.
        Saves and reloads VGG19 replacing MaxPooling with AveragePooling.
        Saves the model in the instance attribute model.
        """
        vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg19.save('VGG19_base_model')

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            'VGG19_base_model', custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            layer.trainable = False
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_output = layer.output

        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer.

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            tf.Tensor of shape (1, c, c) - gram matrix
        """
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        product = int(h * w)
        features = tf.reshape(input_layer, (product, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(product, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        Sets:
            gram_style_features - list of gram matrices from style image
            content_feature - content layer output of content image
        """
        vgg19 = tf.keras.applications.vgg19

        preprocess_style = vgg19.preprocess_input(self.style_image * 255)
        preprocess_content = vgg19.preprocess_input(self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        self.gram_style_features = [
            self.gram_matrix(feature) for feature in style_features]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor of shape (1, h, w, c)
            gram_target: tf.Tensor of shape (1, c, c)

        Returns:
            tf.Tensor - the layer's style cost
        """
        if not (isinstance(style_output, tf.Tensor) or
                isinstance(style_output, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[3]
        if not (isinstance(gram_target, tf.Tensor) or
                isinstance(gram_target, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_generated = self.gram_matrix(style_output)
        cost = tf.reduce_mean(tf.square(gram_generated - gram_target))

        return cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.

        Args:
            style_outputs: list of tf.Tensor style outputs for generated image

        Returns:
            tf.Tensor - the style cost
        """
        l = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != l:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(l))

        weight = 1.0 / l
        style_cost = 0.0
        for i, output in enumerate(style_outputs):
            style_cost += weight * self.layer_style_cost(
                output, self.gram_style_features[i])

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image.

        Args:
            content_output: tf.Tensor - content output for the generated image

        Returns:
            tf.Tensor - the content cost
        """
        s = self.content_feature.shape
        if not (isinstance(content_output, tf.Tensor) or
                isinstance(content_output, tf.Variable)) or \
                content_output.shape != s:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost
