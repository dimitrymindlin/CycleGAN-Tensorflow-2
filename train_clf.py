
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras

import data
from classifiers.apple_orange_model import AppleOrangeModel

TF_LOG_DIR = "logs/apple_orange_clf"
checkpoint_path_name = "checkpoints/inception_apple_orange/"

py.arg('--dataset', default='apple2orange')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=520)  # load image to this size
py.arg('--crop_size', type=int, default=512)  # then crop to this size
py.arg('--batch_size', type=int, default=32)
py.arg('--epochs', type=int, default=20)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--counterfactual_loss_weight', type=float, default=5.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# ==============================================================================
# =                                    data                                    =
# ==============================================================================
special_normalisation = tf.keras.applications.inception_v3.preprocess_input

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')

A_B_dataset, len_dataset = data.make_concat_dataset(A_img_paths[200:], B_img_paths[200:], args.batch_size, args.load_size,
                                                    args.crop_size, training=True, repeat=False,
                                                    special_normalisation=special_normalisation)

A_B_dataset_valid, _ = data.make_concat_dataset(A_img_paths[:200], B_img_paths[:200], args.batch_size, args.load_size,
                                                    args.crop_size, training=True, repeat=False,
                                                    special_normalisation=special_normalisation)



A_B_dataset_test, _ = data.make_concat_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size,
                                               args.crop_size, training=True, repeat=False,
                                               special_normalisation=special_normalisation)

# ==============================================================================

model = AppleOrangeModel(img_shape=(args.crop_size, args.crop_size, 3)).model()

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