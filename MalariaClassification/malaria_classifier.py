"""
Title: malaria_classifier.py

Created on: 2/2/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""

from tqdm import tqdm
from tensorflow.keras.utils import image_dataset_from_directory
import os
import shutil
import random


TRAIN_DIR = os.path.join(os.getcwd(), 'Datasets', 'Train')
TEST_DIR = os.path.join(os.getcwd(), 'Datasets', 'Test')


def reset_folders():
    # Destroy and recreate directories:
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'Infected'))
    os.makedirs(os.path.join(TRAIN_DIR, 'Uninfected'))
    os.makedirs(os.path.join(TEST_DIR, 'Infected'))
    os.makedirs(os.path.join(TEST_DIR, 'Uninfected'))


def split_raw_dataset(test_split: float = 0.2, reset: bool = False):
    # Check if work is already done:
    if reset is False:
        return None
    else:
        # Check total number of images available:
        infected = os.listdir(os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized'))
        uninfected = os.listdir(os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected'))
        total_images = len(infected) + len(uninfected)
        total_test_images = int(total_images * test_split)

        # Randomly select images from the raw dataset:
        random.seed(420)
        random.shuffle(infected)
        train_infected = infected[total_test_images:]
        train_uninfected = uninfected[total_test_images:]
        test_infected = infected[:total_test_images]
        test_uninfected = uninfected[:total_test_images]

        # Build directories:
        reset_folders()
        for img in tqdm(train_infected, desc='Building Training Infected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized', img)
            destination = os.path.join(TRAIN_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(train_uninfected, desc='Building Training Uninfected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected', img)
            destination = os.path.join(TRAIN_DIR, 'Uninfected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_infected, desc='Building Test Infected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized', img)
            destination = os.path.join(TEST_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_uninfected, desc='Building Test Uninfected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected', img)
            destination = os.path.join(TEST_DIR, 'Uninfected', img)
            shutil.copy(source, destination)


def get_train_dataloader(batch_size: int, im_height: int, im_width: int, validation: float = 0.2):
    return image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=validation,
        subset="training",
        seed=420,
        image_size=(im_height, im_width),
        batch_size=batch_size
    )


def main():
    split_raw_dataset()
    return {}


if __name__ == "__main__":
    main()
