import os

import numpy as np
import csv

from zsl.utils import log


def process_classes(path: str, whitelist: str = None, verbose: bool = False):
    log("Starting preprocessing of image-based classes...", verbose=verbose)
    if not os.path.exists(path):
        log(f"The given directory does not exist \"{path}\", but must exist in order to retrieve the necessary training and zero-shot data",
            50)
        return

    whitelisted_classes = retrieve_whitelisted_classes(whitelist)
    actual_classes = retrieve_actual_classes(path, whitelisted_classes)

    train, zsl = split_train_and_zsl(actual_classes, [0.7, 0.3])

    log("The classes have been successfully retrieved and split into training and zero-shot!", 10)

    return train, zsl


def retrieve_actual_classes(path: str, whitelisted_classes: [] = None, verbose: bool = False):
    read_classes = []

    log("Retrieving actual classes...", verbose=verbose)
    with open(f"{path}/classes.txt", newline='') as classes:
        class_reader = csv.reader(classes, delimiter='\t')

        for c in class_reader:
            found_class = c[1]

            if found_class in whitelisted_classes:
                read_classes.append(c[1].strip())

    log(f"The following classes have been retrieved: {read_classes}", 20, verbose=verbose)
    return read_classes


def retrieve_whitelisted_classes(whitelist: str = None, verbose: bool = False):
    whitelisted = []

    log("Reading Whitelisted Classes...", verbose=verbose)
    if whitelist is not None:
        with open(whitelist, newline='') as whitelisted_classes:
            for c in whitelisted_classes:
                whitelisted.append(c.strip())

    log(f"The following whitelisted classes have been retrieved: {whitelisted}", 20, verbose=verbose)
    return whitelisted


def split_train_and_zsl(arr: [], split: [], verbose: bool = False):
    log("Separating classes into training and zero-shot splits...", verbose=verbose)
    train_and_zsl = split_array(arr, split)

    train = train_and_zsl[0]
    zsl = train_and_zsl[1]

    log(f"The following train and zsl split has been made: {train} (training) and {zsl} (ZSL)", 20, verbose=verbose)

    return train, zsl


def split_array(arr: [], split: []):
    a = np.array(arr)
    p = np.array(split)

    return np.split(a, (len(a) * p[:-1].cumsum()).astype(int))
