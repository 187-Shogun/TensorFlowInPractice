"""
Title: image_classification.py

Created on: 1/5/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Classify images of clothing
"""

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Sequential
import numpy as np


def get_train_test_datasets() -> tuple:
    """ Download and load into memory the fashion MNIST dataset from the TF API. """
    return fashion_mnist.load_data()


def get_class_names() -> dict:
    """ Label class names where index corresponds to actual label. """
    return {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }


def plot_image(x: np.array) -> None:
    """ Plot image from array. """
    plt.figure()
    plt.imshow(x)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def plot_images_collage(images: np.array, classes: np.array) -> None:
    """ Plot several images and their labels. """
    plt.figure(figsize=(10, 10))
    for x in range(25):
        plt.subplot(5, 5, x+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[x], cmap=plt.cm.binary)
        plt.xlabel(get_class_names().get(classes[x]))


def plot_prediction(i: int, predictions: np.array, labels: np.array, images: np.array) -> None:
    """ Plot image and the prediction/true labels. """
    true_label, image = labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions[i])
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    a = get_class_names().get(predicted_label)
    b = np.max(predictions) * 100
    c = get_class_names().get(true_label)
    plt.xlabel(f'Predicted: {a}, {b}%, True: {c}', color=color)
    return None


def plot_results(data: dict, metric: str) -> None:
    """ Plot training results. """
    plt.figure(figsize=(13, 21))
    plt.grid(False)
    plt.xticks(np.array(range(len(data.get(metric)))))
    plt.plot(data.get(metric))
    plt.title(metric)
    plt.show()


def normalize_array(x: np.array) -> np.array:
    """ Divide by 255 to limit values between 0 and 1. """
    return x / 255


def build_basic_network() -> Sequential:
    """ Build a basic neural network. """
    model = Sequential([
        Conv2D(filters=3, kernel_size=2, activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def main():
    """ Run script. """
    epochs = 10
    (train_images, train_labels), (test_images, test_labels) = get_train_test_datasets()
    train_images, test_images = normalize_array(train_images), normalize_array(test_images)
    model = build_basic_network()
    training = model.fit(train_images, train_labels, epochs=epochs)
    plot_results(training.history, 'loss')
    plot_results(training.history, 'accuracy')


if __name__ == '__main__':
    main()
