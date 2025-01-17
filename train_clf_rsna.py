import pylib as py
import sklearn
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import data
from classifiers.rsna_model import RSNA_Net

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="log")
np.random.seed(1000)
dimension = 512
batch_size = 16
special_normalisation = tf.keras.applications.inception_v3.preprocess_input
checkpoint_path_name = "checkpoints/inception_rsna"
TF_LOG_DIR = "logs"

########## Data ############


A_img_paths = py.glob(py.join("../tensorflow_datasets/downloads/rsna", 'normal'), '*.jpg')[:8851]
B_img_paths = py.glob(py.join("../tensorflow_datasets/downloads/rsna", 'pneumonia'), '*.jpg')

A_train_paths, A_test_paths = sklearn.model_selection.train_test_split(A_img_paths, test_size=0.2)
B_train_paths, B_test_paths = sklearn.model_selection.train_test_split(B_img_paths, test_size=0.2)
A_train_paths, A_valid_paths = sklearn.model_selection.train_test_split(A_train_paths, test_size=0.1)
B_train_paths, B_valid_paths = sklearn.model_selection.train_test_split(B_train_paths, test_size=0.1)

A_B_dataset, len_dataset = data.make_concat_dataset(A_train_paths, B_train_paths, batch_size, dimension,
                                                    dimension, training=True, repeat=False,
                                                    special_normalisation=special_normalisation)

A_B_dataset_valid, _ = data.make_concat_dataset(A_valid_paths, B_valid_paths, batch_size, dimension,
                                                    dimension, training=True, repeat=False,
                                                    special_normalisation=special_normalisation)

A_B_dataset_test, _ = data.make_concat_dataset(A_test_paths, B_test_paths, batch_size, dimension,
                                               dimension, training=True, repeat=False,
                                               special_normalisation=special_normalisation)

########### Model #############



model = RSNA_Net().model()

my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_name,
                                    # Callback to save the Keras model or model weights at some frequency.
                                    monitor='val_accuracy',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    save_freq='epoch'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                      # Reduce learning rate when a metric has stopped improving.
                                      factor=0.1,
                                      patience=3,
                                      min_delta=0.001,
                                      verbose=1,
                                      min_lr=1e-8),
    keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                histogram_freq=1,
                                write_graph=True,
                                write_images=False,
                                update_freq='epoch',
                                profile_batch=30,
                                embeddings_freq=0,
                                embeddings_metadata=None
                                ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=8,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                                  )
]

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=2, from_logits=False)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc])

# Model Training
history = model.fit(A_B_dataset,
                    epochs=40,
                    verbose=1,
                    class_weight=None,
                    validation_data=A_B_dataset_valid,
                    callbacks=my_callbacks)

print("Train History")
result = model.evaluate(A_B_dataset_test)
result = dict(zip(model.metrics_names, result))
result_matrix = [[k, str(w)] for k, w in result.items()]
for metric, value in zip(model.metrics_names, result):
    print(metric, ": ", value)

print("Result Matrix")
print(result_matrix)
print("Result")
print(result)
model.save(checkpoint_path_name + 'model')