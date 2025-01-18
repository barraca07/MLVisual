import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

def visualise(model, fig_title, dataset="overlapping_blobs"):
    if dataset == "separable_blobs":
        X, y = make_blobs(n_samples=80, centers=2, n_features=2, random_state=1, cluster_std=1.5)
    elif dataset == "overlapping_blobs":
        X, y = make_blobs(n_samples=80, centers=2, n_features=2, random_state=12, cluster_std=3.1)
    elif dataset == "circles":
        X, y = make_circles(80, random_state=5, noise=0.1, factor=0.6)
    elif dataset == "moons":
        X, y = make_moons(80, random_state=2, noise=0.15)
    elif dataset == "spiral":
        X, y = generate_spiral(80)
    else:
        raise Exception("The 'dataset' argument to 'visualise_synthetic' must be one of ['separable_blobs', 'overlapping_blobs', 'circles', 'moons', 'spiral'].")
    
    feature_1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 325)
    feature_2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 325)
    grid_1, grid_2 = np.meshgrid(feature_1, feature_2)
    X_test = np.column_stack([grid_1.ravel(), grid_2.ravel()])
    fit_model = model.fit(X, y)

    prob = fit_model.predict_proba(X_test)
    y_pred_prob = prob[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=120)
    fig.suptitle(fig_title)
    ax1.set_title("Probability Prediction")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=red_blue_gradient(y_pred_prob), s=0.25)
    ax1.scatter(X[:, 0], X[:, 1], c=[(1, 0, 0) if point == 0 else (0, 0, 1) for point in list(y)], edgecolor='k', s=40, linewidth=1)
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    y_pred = fit_model.predict(X_test)
    ax2.set_title("Class Prediction")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=red_blue_gradient(y_pred), s=0.25)
    ax2.scatter(X[:, 0], X[:, 1], c=[(1, 0, 0) if point == 0 else (0, 0, 1) for point in list(y)], edgecolor='k', s=40)
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.show()

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