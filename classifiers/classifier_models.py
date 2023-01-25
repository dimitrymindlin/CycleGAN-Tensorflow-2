import tensorflow as tf
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential


class Domain2DomainModel(tf.keras.Model):

    def __init__(self, weights='imagenet', img_shape=(512, 512, 3)):
        super(Domain2DomainModel, self).__init__(name='Domain2DomainModel')
        self.weight_regularisation = regularizers.l2(0.002)
        self._input_shape = img_shape
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            input_shape=self._input_shape,
                                                            input_tensor=self.img_input,
                                                            weights=weights,
                                                            pooling='avg',
                                                            classes=2)
        self.base_model.trainable = True
        self.classifier = tf.keras.layers.Dense(2, activation="softmax", name="predictions")

    def call(self, x):
        x = self.base_model(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)


class CatsVSDogsModel(tf.keras.Model):
    def __init__(self, img_shape=(512, 512, 3)):
        super(CatsVSDogsModel, self).__init__(name='CatsVSDogsModel')
        self.img_shape = img_shape

    def model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        return model
