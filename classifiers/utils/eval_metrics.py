import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

from classifiers.utils.training_utils import get_labels_from_tfds


def log_and_print_evaluation(model, data, config, file_writer=None):
    result = model.evaluate(data.A_B_dataset_test)
    result = dict(zip(model.metrics_names, result))
    result_matrix = [[k, str(w)] for k, w in result.items()]
    print(result_matrix)

    m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=True)
    y_pred = model.predict(data.A_B_dataset_test)

    y_predicted = np.argmax(y_pred, axis=1)
    y_true = get_labels_from_tfds(data.A_B_dataset_test)
    print(classification_report(y_true, y_predicted))
    print(y_predicted.shape, y_true.shape)
    print(confusion_matrix(y_true, y_predicted))
    m.update_state(y_true, y_predicted)
    print('TFA Kappa score result: ', m.result().numpy())

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_predicted)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    result_matrix.append(["TFA Kappa score", str(m.result().numpy())])
    result_matrix.append(["TN", str(tn)])
    result_matrix.append(["FP", str(fp)])
    result_matrix.append(["FN", str(fn)])
    result_matrix.append(["TP", str(tp)])
    result_matrix.append(["Precision", str(precision)])
    result_matrix.append(["Recall", str(recall)])
    result_matrix.append(["F1", str(f1_score)])
    try:
        with file_writer.as_default():
            tf.summary.text(f"{config['model']['name']}_evaluation", tf.convert_to_tensor(result_matrix), step=0)
    except AttributeError:
        pass

    print("Result matrix")
    print(result_matrix)

    print("Classification Report")

    print(classification_report(y_true, y_predicted))
