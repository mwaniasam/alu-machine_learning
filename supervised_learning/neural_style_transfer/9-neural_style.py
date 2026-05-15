#!/usr/bin/env python3
"""Module for Neural Style Transfer - Task 9: Generate Image"""
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
        vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg19.trainable = False

        inputs = tf.keras.Input(shape=(None, None, 3))
        x = inputs
        for layer in vgg19.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)(x)
            else:
                layer.trainable = False
                x = layer(x)

        new_model = tf.keras.Model(inputs=inputs, outputs=x)

        outputs = []
        for name in self.style_layers:
            outputs.append(new_model.get_layer(name).output)
        outputs.append(new_model.get_layer(self.content_layer).output)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

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

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor of shape (1, h, w, c) - layer style output
                          of the generated image
            gram_target: tf.Tensor of shape (1, c, c) - gram matrix of the
                         target style output for that layer

        Returns:
            tf.Tensor - the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[3]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                len(gram_target.shape) != 3 or
                gram_target.shape[1] != c or
                gram_target.shape[2] != c):
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
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image.

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3)

        Returns:
            (J, J_content, J_style) - total, content, and style costs
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = list(outputs[:-1])
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image.

        Args:
            generated_image: tf.Tensor or tf.Variable of shape (1, nh, nw, 3)

        Returns:
            (gradients, J_total, J_content, J_style)
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        with tf.GradientTape() as tape:
            J, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J, generated_image)

        return gradients, J, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        Generates the neural style transferred image.

        Args:
            iterations: int - number of iterations for gradient descent
            step: int or None - print info every this many steps
            lr: float - learning rate
            beta1: float - Adam beta1 parameter
            beta2: float - Adam beta2 parameter

        Returns:
            (generated_image, cost) - best generated image and its cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.contrib.eager.Variable(self.content_image)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J, J_content, J_style = self.compute_grads(generated_image)
            if J < best_cost:
                best_cost = J.numpy()
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}".format(
                    i, J, J_content, J_style))

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0, 1))

        best_image = best_image[0]

        return best_image, best_cost
