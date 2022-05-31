import tensorflow as tf
from keras import regularizers


class Domain2DomainModel(tf.keras.Model):

    def __init__(self, weights='imagenet', img_shape=(512, 512, 3)):
        super(Domain2DomainModel, self).__init__(name='Domain2DomainModel')
        #self._input_shape = img_shape
        #self.img_input = tf.keras.Input(shape=self._input_shape)
        self.img_input = tf.keras.Input(shape=(256, 256, 3))
        self.base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                         weights=weights,
                                                                         pooling='avg',
                                                                         classes=2)
        self.base_model.trainable = True
        self.classifier = tf.keras.layers.Dense(2, activation="softmax", name="predictions")

    def call(self, x):
        x = self.img_input(x)
        x = tf.keras.layers.Resizing(512, 512)(x)
        x = self.base_model(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.img_input
        x = tf.keras.layers.Resizing(512, 512)(x)
        x = self.base_model(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)


def get_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Resizing(512, 512))
    model.add(tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                         weights="imagenet",
                                                                         pooling='avg',
                                                                         classes=2))
    model.add(tf.keras.layers.Dense(2, activation="softmax", name="predictions"))
    return model