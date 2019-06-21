"""Run all scripts in this subrepository."""
import sys

import sklearn
import numpy as np

import balancing
import augmentation
import preprocessing
import symbol_recog

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Skip flags: {}".format(sys.argv[1:]))

    if {"b", "a", "p"}.isdisjoint(sys.argv):
        print("Balancing classes...")
        balancing.balance_classes("./data-original", random_seed=1234)

    else:
        print("Skipped data balancing.")

    if {"a", "p"}.isdisjoint(sys.argv):
        print("Augmenting data...")
        augmentation.augment_data("./data-balanced", random_seed=2805)

    else:
        print("Skipped data augmenting.")

    if "p" not in sys.argv:
        print("Preprocessing augmented data...")
        preprocessing.preprocess("./data-augmented")

    else:
        print("Skipped data preprocessing.")

    print("Getting data...")
    X, y = symbol_recog.get_data("./data-augmented-preprocessed")
    print("Got all data (total of {} instances).".format(y.shape[0]))

    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    model = symbol_recog.CNNModel(
        random_seed=1234,
        num_classes=np.unique(y).size)
    symbol_recog.train_models([3, 16, 25, 26, 27])

    print("Process finished.")
