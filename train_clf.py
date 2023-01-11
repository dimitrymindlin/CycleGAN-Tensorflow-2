import json
import os
from datetime import datetime

import numpy as np
from rsna.tfds_from_disc import get_rsna_ds_for_clf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_addons as tfa
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras

import standard_datasets_loading
from classifiers.classifier_models import Domain2DomainModel, CatsVSDogsModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
dataset = "horse2zebra"
img_size = 512
clf_name = "alexnet"
TF_LOG_DIR = f"logs/{dataset}_clf_{img_size}/{execution_id}"
TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"

py.arg('--dataset', default=dataset)
py.arg('--datasets_dir', default='tensorflow_datasets')
py.arg('--load_size', type=int, default=img_size + 30)  # load image to this size
py.arg('--crop_size', type=int, default=img_size)  # then crop to this size
py.arg('--batch_size', type=int, default=32)
py.arg('--epochs', type=int, default=40)
py.arg('--load_checkpoint', type=str, default=None)
args = py.args()

# ==============================================================================
# =                                    data                                    =
# ==============================================================================
special_normalisation = tf.keras.applications.inception_v3.preprocess_input

if args.dataset in ["horse2zebra", "apple2orange"]:
    A_img_paths = py.glob(py.join(TFDS_PATH, args.dataset, 'trainA'), '*.jpg')
    B_img_paths = py.glob(py.join(TFDS_PATH, args.dataset, 'trainB'), '*.jpg')
    A_img_paths_test = py.glob(py.join(TFDS_PATH, args.dataset, 'testA'), '*.jpg')
    B_img_paths_test = py.glob(py.join(TFDS_PATH, args.dataset, 'testB'), '*.jpg')

    A_B_dataset, len_dataset = standard_datasets_loading.make_concat_dataset(A_img_paths[200:], B_img_paths[200:],
                                                                             args.batch_size, args.load_size,
                                                                             args.crop_size, training=True,
                                                                             repeat=False,
                                                                             special_normalisation=special_normalisation)

    A_B_dataset_valid, _ = standard_datasets_loading.make_concat_dataset(A_img_paths[:200], B_img_paths[:200],
                                                                         args.batch_size, args.load_size,
                                                                         args.crop_size, training=True, repeat=False,
                                                                         special_normalisation=special_normalisation)

    A_B_dataset_test, _ = standard_datasets_loading.make_concat_dataset(A_img_paths_test, B_img_paths_test,
                                                                        args.batch_size,
                                                                        args.load_size,
                                                                        args.crop_size, training=True, repeat=False,
                                                                        special_normalisation=special_normalisation)
elif args.dataset == "rsna":
    A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train = get_rsna_ds_for_clf(TFDS_PATH,
                                                                                              args.batch_size,
                                                                                              args.crop_size,
                                                                                              args.load_size,
                                                                                              special_normalisation,
                                                                                              channels=1)
else:  # Mura
    pass


# ==============================================================================

def evaluate_model(model):
    print("Evaluating model")
    y_true = []
    y_pred = []
    for i, (x, y) in enumerate(A_B_dataset_test):
        y_true.append(y.numpy())
        y_pred.append(model.predict(x).argmax(axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"F1: {f1_score(y_true, y_pred)}")


if args.load_checkpoint is not None:
    checkpoint_path_name = f"checkpoints/{clf_name}_{dataset}/{args.load_checkpoint}/"
    model = tf.keras.models.load_model(
        f"{ROOT_DIR}/{checkpoint_path_name}/model", compile=False)
    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=2, from_logits=False)
    metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    metric_precision = tf.keras.metrics.Precision()
    metric_recall = tf.keras.metrics.Recall()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc, metric_f1, metric_precision, metric_recall])
    result = model.evaluate(A_B_dataset_test)
    result = dict(zip(model.metrics_names, result))
    print(json.dumps(result))
    quit()
else:
    if dataset in ["horse2zebra", "apple2orange"]:
        model = CatsVSDogsModel(img_shape=(args.crop_size, args.crop_size, 3)).model()
    else:
        model = Domain2DomainModel(img_shape=(args.crop_size, args.crop_size, 3)).model()

    checkpoint_path_name = f"checkpoints/{clf_name}_{dataset}/{execution_id}/"

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
                    epochs=args.epochs,
                    verbose=1,
                    class_weight=None,
                    validation_data=A_B_dataset_valid,
                    callbacks=my_callbacks)

print("Train History")
result = model.evaluate(A_B_dataset_test)
result = dict(zip(model.metrics_names, result))
for metric, value in result.items():
    print(metric, ": ", value)

model.save(checkpoint_path_name + 'model')
