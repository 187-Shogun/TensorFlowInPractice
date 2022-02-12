"""
Title: cars_classifier.py

Created on: 2/9/2022

Author: 187-Shogun

Encoding: UTF-8



"""


from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras.applications import vgg16
from tensorflow import image
from pytz import timezone
from datetime import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
import os


plt.style.use('dark_background')
AUTOTUNE = tf.data.AUTOTUNE
LOGS_DIR = os.path.join(os.getcwd(), 'Logs')
CM_DIR = os.path.join(os.getcwd(), 'ConfusionMatrixes')
MODELS_DIR = os.path.join(os.getcwd(), 'Models')
BATCH_SIZE = 32
IM_HEIGHT = 224
IM_WIDTH = 224
EPOCHS = 10000
PATIENCE = 50
RANDOM_SEED = 420


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def download_dataset() -> tuple:
    """ Get dataset from the TFDS library. """
    train, info = tfds.load(
        'cars196',
        shuffle_files=True,
        batch_size=BATCH_SIZE,
        with_info=True,
        as_supervised=True,
        split=['train']
    )
    val = tfds.load(
        'cars196',
        shuffle_files=True,
        batch_size=BATCH_SIZE,
        as_supervised=True,
        split=['test[50%:]']
    ),
    test = tfds.load(
        'cars196',
        shuffle_files=True,
        batch_size=BATCH_SIZE,
        as_supervised=True,
        split=['test[:50%]']
    )
    return train[0], val[0][0], test[0], info


def get_baseline_nn(num_classes: int):
    """ Build a Sequential NN model and compile it. """
    model_layers = [
        layers.Flatten(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        layers.Dense(100, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ]
    model = models.Sequential(model_layers, name='BaselineModel')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_cnn(num_classes: int):
    """ Build a Sequential CNN model and compile it. """
    model_layers = [
        layers.InputLayer(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        layers.RandomFlip(),
        layers.RandomRotation(0.3),
        layers.RandomContrast(0.3),
        layers.Conv2D(128, 3, activation='relu', strides=2, kernel_initializer='he_uniform'),
        layers.Conv2D(64, 3, activation='relu', strides=2, kernel_initializer='he_uniform'),
        layers.Conv2D(64, 3, activation='relu', strides=2, kernel_initializer='he_uniform'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(num_classes, activation='softmax')
    ]
    model = models.Sequential(model_layers, name='CustomModel-CNN')
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_pretrained_vgg16(n_classes: int):
    """ Load a pretrained VGG16 model and add a custom laer on top. """
    base_model = vgg16.VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    ga_layer = layers.GlobalAveragePooling2D()(base_model.output)
    out_layer = layers.Dense(n_classes, activation='softmax')(ga_layer)
    model = Model(inputs=base_model.inputs, outputs=out_layer)
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )
    return model


def main():
    """ Run script. """
    X_train, X_val, X_test, info = download_dataset()
    label_names = info.features['label'].names

    # Normalize data prior training:
    normalization_layer = layers.Rescaling(1./255.)
    X_train = X_train.map(lambda x, y: (image.resize(x, [IM_HEIGHT, IM_WIDTH]), y))
    X_train = X_train.map(lambda x, y: (normalization_layer(x), y))
    X_train = X_train.prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.map(lambda x, y: (image.resize(x, [IM_HEIGHT, IM_WIDTH]), y))
    X_val = X_val.map(lambda x, y: (normalization_layer(x), y))
    X_val = X_val.prefetch(buffer_size=AUTOTUNE)

    # Start training a single model:
    model = get_pretrained_vgg16(len(label_names))
    version_name = get_model_version_name(model.name)
    tb_logs = callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    # lr_scheduler = callbacks.ReduceLROnPlateau(factor=.5, patience=5)
    model.fit(X_train, validation_data=X_val, epochs=EPOCHS, callbacks=[tb_logs, early_stop])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return {}


if __name__ == "__main__":
    main()
