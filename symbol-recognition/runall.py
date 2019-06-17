"""Run all scripts in this subrepository."""
import balancing
import augmentation
import preprocessing

if __name__ == "__main__":
    print("Balancing classes...")
    balancing.balance_classes("./data-original", random_seed=1234)

    print("Augmenting data...")
    augmentation.augment_data("./data-balanced", random_seed=2805)

    print("Preprocessing augmented data...")
    preprocessing.preprocess("./data-augmented")

    print("Process finished.")
