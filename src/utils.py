from ucimlrepo import fetch_ucirepo
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def map_class_labels(y_train):
    transformed_y_train = y_train.copy()
    classes = sorted(list(np.unique(transformed_y_train)))
    for idx, label in enumerate(classes):
        transformed_y_train[transformed_y_train == label] = idx
    transformed_y_train = transformed_y_train.astype(int)
    return transformed_y_train, classes

def load_abalone_data(train_proportion = 0.7) -> tuple[NDArray, NDArray, NDArray, NDArray]:

    abalone = fetch_ucirepo(id=1) 
    
    X = abalone.data.features
    X = X.drop(["Sex"], axis=1)
    y = abalone.data.targets

    nearest_int = int(X.shape[0]*train_proportion)
    
    return X[:nearest_int].values, X[nearest_int:].values, y[:nearest_int].values.reshape(-1), y[nearest_int:].values.reshape(-1)

def generate_synthetic_data(centers):
    return make_blobs(n_samples=80, centers=2, n_features=centers, random_state=12, cluster_std=3.1)

def red_blue_gradient(y_pred):
    result = []
    for point in list(y_pred):
        if point <= 0.5:
            col = (1, 0.25+3/2*point, 0.25+3/2*point)
        else:
            col = (1.75-3/2*point, 1.75-3/2*point, 1)
        result.append(col)
    return result

def visualise_synthetic(model, title):
    X, y = generate_synthetic_data(centers=2)
    feature_1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 325)
    feature_2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 325)
    grid_1, grid_2 = np.meshgrid(feature_1, feature_2)
    X_test = np.column_stack([grid_1.ravel(), grid_2.ravel()])
    fit_model = model.fit(X, y)

    prob = fit_model.predict_proba(X_test)
    y_pred_prob = prob[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=120)
    fig.suptitle(title)
    ax1.set_title("Probability Prediction")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=red_blue_gradient(y_pred_prob), s=0.25)
    ax1.scatter(X[:, 0], X[:, 1], c=[(1, 0, 0) if point == 0 else (0, 0, 1) for point in list(y)], edgecolor='k', s=40, linewidth=1)
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_aspect('equal', adjustable='box')

    y_pred = fit_model.predict(X_test)
    ax2.set_title("Class Prediction")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=red_blue_gradient(y_pred), s=0.25)
    ax2.scatter(X[:, 0], X[:, 1], c=[(1, 0, 0) if point == 0 else (0, 0, 1) for point in list(y)], edgecolor='k', s=40)
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_aspect('equal', adjustable='box')

    plt.show()