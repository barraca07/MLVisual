import numpy as np

def map_class_labels(y_train):
    transformed_y_train = y_train.copy()
    classes = sorted(list(np.unique(transformed_y_train)))
    for idx, label in enumerate(classes):
        transformed_y_train[transformed_y_train == label] = idx
    transformed_y_train = transformed_y_train.astype(int)
    return transformed_y_train, classes

def generate_spiral(n_points):
    np.random.seed(0)
    X = []
    y = []
    for class_number in range(2):
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(class_number * np.pi, class_number * np.pi + 2 * np.pi, n_points) + np.random.randn(n_points) * 0.5
        dx = r * np.sin(t)
        dy = r * np.cos(t)
        X.extend(np.c_[dx, dy])
        y.extend([class_number] * n_points)
    X = np.array(X)
    y = np.array(y)
    return X, y

def red_blue_gradient(y_pred):
    result = []
    for point in list(y_pred):
        if point <= 0.5:
            col = (1, 0.25+3/2*point, 0.25+3/2*point)
        else:
            col = (1.75-3/2*point, 1.75-3/2*point, 1)
        result.append(col)
    return result