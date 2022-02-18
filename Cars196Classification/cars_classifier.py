"""
Title: cars_classifier.py

Created on: 2/9/2022

Author: 187-Shogun

Encoding: UTF-8
"""


from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import applications
from tensorflow import image
from pytz import timezone
from datetime import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
import os


# Global variables:
AUTOTUNE = tf.data.AUTOTUNE
LOGS_DIR = os.path.join(os.getcwd(), 'Logs')
CM_DIR = os.path.join(os.getcwd(), 'ConfusionMatrixes')
MODELS_DIR = os.path.join(os.getcwd(), 'Models')
BATCH_SIZE = 16
IM_HEIGHT = 224
IM_WIDTH = 224
EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 420


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def download_dataset() -> tuple:
    """ Get dataset from the TFDS library. """
    train_a, info = tfds.load(
        'cars196',
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        split=['train']
    )
    a, b, c = tfds.even_splits('test', n=3, drop_remainder=True)
    train_b, val, test = tfds.load(
        'cars196',
        shuffle_files=True,
        as_supervised=True,
        split=[a, b, c]
    )
    # Unpack elements:
    train = train_a[0].concatenate(train_b)
    return train, val, test, info


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


def get_pretrained_network(n_classes: int):
    """ Load a pretrained model and add a custom laer on top. """
    # Import pretrained lower layers:
    input_shape = (IM_HEIGHT, IM_WIDTH, 3)
    base_model = applications.xception.Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers:
    pooling_layer = layers.GlobalAveragePooling2D()(base_model.output)
    out_layer = layers.Dense(n_classes, activation='softmax')(pooling_layer)
    model = Model(inputs=base_model.inputs, outputs=out_layer)
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )
    return model


def train_pretrained_network(ds_info, training_ds, validation_ds, pretrain_rounds=10):
    """ Build a model from a pretrained one. Freeze the bottom layers and train the top layers
    for n epochs. Then, unfreeze the bottom layers, adjust the learning rate down and train the
    model one more time. """
    # Import pretrained lower layers:
    label_names = ds_info.features['label'].names
    input_shape = (IM_HEIGHT, IM_WIDTH, 3)
    base_model = applications.resnet_v2.ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers:
    pooling_layer = layers.GlobalAveragePooling2D()(base_model.output)
    dropout_layer = layers.Dropout(0.25)(pooling_layer)
    out_layer = layers.Dense(len(label_names), activation='softmax')(dropout_layer)
    model = Model(inputs=base_model.inputs, outputs=out_layer, name='ResNet152V2-CNN')
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )

    # Start pretraining:
    version_name = get_model_version_name(model.name)
    tb_logs = callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    model.fit(training_ds, validation_data=validation_ds, epochs=pretrain_rounds, callbacks=[tb_logs])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Unfreeze layers and train the entire network:
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics='accuracy'
    )
    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def get_preprocessing_layer():
    """ Stack preprocessing layers together. """
    return models.Sequential([
        layers.RandomFlip(),
        layers.RandomRotation(0.3),
        layers.RandomContrast(0.3)
    ])


def main():
    """ Run script. """
    # Normalize data prior training:
    X_train, X_val, X_test, info = download_dataset()
    method = image.ResizeMethod.NEAREST_NEIGHBOR
    normalization_layer = layers.Rescaling(1./255.)
    preprocessing_layer = get_preprocessing_layer()
    X_train = X_train.map(lambda x, y: (image.resize(x, [IM_HEIGHT, IM_WIDTH], method=method), y))
    X_train = X_train.map(lambda x, y: (normalization_layer(x), y))
    X_train = X_train.batch(BATCH_SIZE).map(lambda x, y: (preprocessing_layer(x), y)).prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.map(lambda x, y: (image.resize(x, [IM_HEIGHT, IM_WIDTH], method=method), y))
    X_val = X_val.map(lambda x, y: (normalization_layer(x), y))
    X_val = X_val.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    # Start training a single model:
    model = train_pretrained_network(info, X_train, X_val)
    return model


if __name__ == "__main__":
    main()
