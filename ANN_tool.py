import itertools
import os
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

warnings.simplefilter('ignore')


def load_image_data(folder, target_size=(256, 256)):
    _inputs = []
    _outputs = []

    for filename in os.listdir(folder):
        source_file = os.path.join(folder, filename)
        if os.path.isfile(source_file) and any(
                filename.lower().endswith(image_extension) for image_extension in ['.jpg', '.jpeg', '.png', '.gif']):
            if folder == 'normal':
                _outputs.append(0)
            elif folder == 'sepia':
                _outputs.append(1)

            image = cv2.imread(source_file)
            resized_image = cv2.resize(image, target_size)
            _inputs.append(resized_image)

    return _inputs, _outputs


def plot_histogram(_outputs, title):
    plt.hist(_outputs, bins=[-0.5, 0.5, 1.5], rwidth=0.8, align='mid')
    plt.xticks([0, 1])
    plt.title(title)
    plt.show()


def split_data(_inputs, _outputs, ratio=0.8):
    data_size = len(_inputs)
    indices = np.random.permutation(data_size)
    train_size = int(data_size * ratio)
    train_indices, _test_indices = indices[:train_size], indices[train_size:]
    _train_inputs, _test_inputs = np.array([_inputs[i] for i in train_indices]), np.array(
        [_inputs[i] for i in _test_indices])
    _train_outputs, _test_outputs = np.array([_outputs[i] for i in train_indices]), np.array(
        [_outputs[i] for i in _test_indices])
    return _train_inputs, _train_outputs, _test_inputs, _test_outputs


def flatten_images(data):
    return np.array(data).reshape(len(data), -1)


def normalize_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_stacked = train_data.reshape(len(train_data), -1)
    test_data_stacked = test_data.reshape(len(test_data), -1)

    scaler.fit(train_data_stacked)
    normalized_train_data = scaler.transform(train_data_stacked)
    normalized_test_data = scaler.transform(test_data_stacked)

    return normalized_train_data, normalized_test_data


def eval_multi_class(real_labels, computed_labels):
    conf_matrix = confusion_matrix(real_labels, computed_labels)
    acc = np.mean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    _precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    _recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return acc, _precision, _recall, conf_matrix


def plot_confusion_matrix(cm, class_names, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    normal_inputs, normal_outputs = load_image_data('normal')
    sepia_inputs, sepia_outputs = load_image_data('sepia')

    _output_names = ['normal', 'sepia']

    inputs = normal_inputs + sepia_inputs
    outputs = normal_outputs + sepia_outputs

    plot_histogram(outputs, 'Output Distribution')

    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)

    print("Shape of train_inputs:", train_inputs.shape)
    print("Shape of test_inputs:", test_inputs.shape)

    train_inputs = flatten_images(train_inputs)
    test_inputs = flatten_images(test_inputs)

    train_inputs, test_inputs = normalize_data(train_inputs, test_inputs)

    # Accuracy: 0.8564102564102565
    # Precision: [0.86842105 0.84415584]
    # Recall: [0.84615385 0.86666667]
    # classifier = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', max_iter=200, solver='sgd',
    #                            verbose=10, random_state=1, learning_rate_init=.003)

    # Accuracy: 0.8692754613807245
    # Precision: [0.86842105 0.87012987]
    # Recall: [0.86842105 0.87012987]
    # classifier = MLPClassifier(hidden_layer_sizes=(15,), activation='relu', max_iter=100, solver='sgd',
    #                            verbose=10, random_state=1, learning_rate_init=.01)

    # Accuracy: 0.8750443105281815
    # Precision: [0.90909091 0.83076923]
    # Recall: [0.87912088 0.87096774]
    # classifier = MLPClassifier(hidden_layer_sizes=(15,), activation='relu', max_iter=200, solver='sgd',
    #                            verbose=10, random_state=1, learning_rate_init=.003)
    #
    # Accuracy: 0.9412820512820512
    # Precision: [0.94805195 0.93421053]
    # Recall: [0.93589744 0.94666667]
    classifier = MLPClassifier(hidden_layer_sizes=(15,), activation='relu', max_iter=200, solver='sgd',
                               verbose=10, random_state=1, learning_rate_init=.001)
    classifier.fit(train_inputs, train_outputs)

    predicted_labels = classifier.predict(test_inputs)

    accuracy, precision, recall, confusion_mat = eval_multi_class(test_outputs, predicted_labels)

    plot_confusion_matrix(confusion_mat, _output_names, "sepia classification")
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)

    n = 6
    m = 5
    fig, axes = plt.subplots(n, m, figsize=(10, 10))
    fig.tight_layout()
    for i in range(0, n):
        for j in range(0, m):
            image = test_inputs[m * i + j].reshape(256, 256, 3)
            axes[i][j].imshow(image)
            if np.any(test_inputs[m * i + j] == test_inputs[m * i + j]):
                font = 'normal'
            else:
                font = 'bold'
            axes[i][j].set_title(
                'real ' + str(test_outputs[m * i + j]) + '\npredicted ' + str(predicted_labels[m * i + j]),
                fontweight=font)
            axes[i][j].set_axis_off()

    plt.show()
