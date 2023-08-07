from tensorflow import keras
import tensorflow as tf
from keras import regularizers
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from classifiers.utils.eval_metrics import log_and_print_evaluation

from mura_data.data.load_mura import get_mura_ds_by_body_part
from classifiers.utils.training_utils import get_labels_from_tfds

from classifiers.MURA.mura_config import mura_clf_training_config
from classifiers.utils.model_utils import get_input_shape_from_config, get_model_by_name
from global_config import ROOT_DIR


class MuraDataset:
    def __init__(self, config):
        self.config = config
        self.special_normalisation = tf.keras.applications.inception_v3.preprocess_input
        self.A_B_dataset, self.A_B_dataset_val, self.A_B_dataset_test, self.len_dataset_train = get_mura_ds_by_body_part(
            'XR_WRIST',
            config["data"]["data_root"],
            config["train"][
                "batch_size"],
            config["data"][
                "image_height"],
            config["data"][
                "image_height"],
            special_normalisation=self.special_normalisation,
            transformed=config["dataset"]["transformed"])
        self.train_y = get_labels_from_tfds(self.A_B_dataset)


class MuraModel(tf.keras.Model):
    "MuraNet Model Class with various base models"

    def __init__(self, config, weights='imagenet'):
        super(MuraModel, self).__init__(name='WristPredictNet')
        self.config = config
        self.weight_regularisation = regularizers.l2(config["train"]["weight_regularisation"]) if config["train"][
            "weight_regularisation"] else None
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = get_model_by_name(self.config, self._input_shape, weights, self.img_input)
        self.base_model.trainable = self.config['train']['train_base']
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="softmax",
                                                name="predictions")

    def call(self, x):
        x = self.base_model(x)
        if self.config["train"]["additional_last_layers"]:
            for layer_count in range(self.config["train"]["additional_last_layers"]):
                print("Adding additional layers...")
                x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
                x = tf.keras.layers.Dropout(self.config["train"]["dropout_value"])(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        if self.config["train"]["additional_last_layers"]:
            for layer_count in range(self.config["train"]["additional_last_layers"]):
                print("Adding additional layers...")
                x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
                x = tf.keras.layers.Dropout(self.config["train"]["dropout_value"])(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)


def train_model(config):
    # Get Settings and set names and paths
    MODEL_NAME = config["model"]["name"]
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
    TF_LOG_DIR = f'tensorboard_logs/logs_training/{MODEL_NAME}/' + TIMESTAMP + "/"
    checkpoint_path_name = f'checkpoints/{MODEL_NAME}/' + TIMESTAMP + '/cp.ckpt'
    checkpoint_path = f'checkpoints/{MODEL_NAME}/' + TIMESTAMP + '/'
    file_writer = tf.summary.create_file_writer(TF_LOG_DIR)

    # Tensorboard config matrix
    config_matrix = [[k, str(w)] for k, w in config["train"].items()]
    with file_writer.as_default():
        tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

    # Load data and class weights
    mura_data = MuraDataset(config)
    if config["train"]["use_class_weights"]:
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(mura_data.train_y),
                                             y=mura_data.train_y)
        d_class_weights = dict(zip(np.unique(mura_data.train_y), class_weights))
    else:
        d_class_weights = None

    # Callbacks
    my_callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_name,
                                        # Callback to save the Keras model or model weights at some frequency.
                                        monitor='val_accuracy',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_freq='epoch'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                          # Reduce learning rate when a metric has stopped improving.
                                          factor=config["train"]["factor_learning_rate"],
                                          patience=config["train"]["patience_learning_rate"],
                                          min_delta=0.001,
                                          verbose=1,
                                          min_lr=config["train"]["min_learning_rate"]),
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
                                      patience=config["train"]["early_stopping_patience"],
                                      mode="max",
                                      baseline=None,
                                      restore_best_weights=True,
                                      )
    ]

    # Load model and set train params and metrics
    model = MuraModel(config).model()

    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                      from_logits=False)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["train"]["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc])

    # Model Training
    history = model.fit(mura_data.A_B_dataset,
                        epochs=config["train"]["epochs"],
                        verbose=1,
                        class_weight=d_class_weights,
                        validation_data=mura_data.A_B_dataset_val,
                        callbacks=my_callbacks)

    # Save whole model
    model.save(checkpoint_path + 'model')

    # Evaluation
    print("Train History")
    print(history)
    print(f"Test Evaluation for {TIMESTAMP}")
    log_and_print_evaluation(model, mura_data, config, file_writer)

    return TIMESTAMP


if __name__ == "__main__":
    config = mura_clf_training_config
    config["model"]["name"] = "inception"

    # TODO: Set correct TFDS path

    TFDS_PATH = ROOT_DIR + "/" + "../tensorflow_datasets/"
    config["data"]["data_root"] = TFDS_PATH
    train_model(config)
