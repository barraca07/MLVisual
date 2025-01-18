import numpy as np

def map_class_labels(y_train):
    transformed_y_train = y_train.copy()
    classes = sorted(list(np.unique(transformed_y_train)))
    for idx, label in enumerate(classes):
        transformed_y_train[transformed_y_train == label] = idx
    transformed_y_train = transformed_y_train.astype(int)
    return transformed_y_train, classes