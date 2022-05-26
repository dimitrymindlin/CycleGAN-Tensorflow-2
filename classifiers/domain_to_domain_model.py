import tensorflow as tf
from keras import regularizers


class Domain2DomainModel(tf.keras.Model):

    def __init__(self, weights='imagenet', img_shape=(512, 512, 3)):
        super(Domain2DomainModel, self).__init__(name='Domain2DomainModel')
        self._input_shape = img_shape
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = tf.keras.applications.MobileNet(include_top=False,
                                                            input_shape=self._input_shape,
                                                            input_tensor=self.img_input,
                                                            weights=weights,
                                                            pooling='avg',
                                                            classes=2)
        self.base_model.trainable = True
        self.classifier = tf.keras.layers.Dense(2, activation="softmax", name="predictions")

    def call(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)