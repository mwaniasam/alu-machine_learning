#!/usr/bin/env python3
"""Module for Neural Style Transfer - Task 3: Extract Features"""
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
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
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
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image = np.expand_dims(image, axis=0)
        scaled = tf.image.resize_bicubic(image, [new_h, new_w])
        scaled = scaled / 255.0
        scaled = tf.clip_by_value(scaled, 0, 1)

        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as base.
        MaxPooling layers are replaced with AveragePooling layers.
        The model outputs style layer outputs followed by content layer output.
        Saves the model in the instance attribute model.
        """
        vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet')
        vgg19.trainable = False

        # Rebuild graph with AveragePooling replacing MaxPooling
        x = vgg19.input
        for layer in vgg19.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)(x)
            else:
                x = layer(x)

        # Build a model that outputs everything up to the last needed layer
        full_model = tf.keras.models.Model(inputs=vgg19.input, outputs=x)

        outputs = []
        for name in self.style_layers:
            outputs.append(full_model.get_layer(name).output)
        outputs.append(full_model.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(
            inputs=vgg19.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer.

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            tf.Tensor of shape (1, c, c) - gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        F = tf.reshape(input_layer, (1, -1, c))
        gram = tf.matmul(F, F, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        Sets:
            gram_style_features - list of gram matrices from style image
            content_feature - content layer output of content image
        """
        vgg19 = tf.keras.applications.vgg19

        style_preprocessed = vgg19.preprocess_input(self.style_image * 255)
        style_outputs = self.model(style_preprocessed)

        content_preprocessed = vgg19.preprocess_input(
            self.content_image * 255)
        content_outputs = self.model(content_preprocessed)

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]
