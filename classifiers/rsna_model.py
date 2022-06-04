# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf
from keras import regularizers


class RSNA_Net(tf.keras.Model):
    "MuraNet Model Class with various base models"

    def __init__(self, weights='imagenet'):
        super(RSNA_Net, self).__init__(name='RSNA_Net')
        self.weight_regularisation = regularizers.l2(0.1)
        self._input_shape = (512, 512, 3)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            input_shape=self._input_shape,
                                                            input_tensor=self.img_input,
                                                            weights=weights,
                                                            pooling='avg',
                                                            classes=2)
        self.classifier = tf.keras.layers.Dense(2, activation="softmax", name="predictions")

    def call(self, x):
        x = self.base_model(x)
        print("Adding additional layers...")
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
