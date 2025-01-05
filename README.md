# MLVisual: Machine Learning Algorithm Visualiser

This repository contains concise, prototype NumPy implementations of common machine learning classification algorithms and a tool for visualising the performance of the algorithms using a simple dataset with two input features. The aim is to provide a resource for facilitating an intuitive understanding of how these algorithms work and how they can be practically implemented in code.

The algorithms currently implemented are:
* Logistic Regression
* K-Nearest Neighbours
* Naive Bayes
* Decision Tree
* Random Forest
* AdaBoost
* Support Vector Machine
* Neural Network

## Visualiser Usage

The `visualise_synthetic` function in `utils.py` allows algorithm probability and class predictions to be visualised using a synthetic 2D binary classification dataset. It is used as follows:

----

**`visualise_synthetic(model, title)`** - Create the model visualisation

Parameters:

`model` (LogisticRegression | NearestNeightbours | NaiveBayes | DecisionTree | RandomForest | AdaBoost | SupportVectorMachine | NeuralNetwork)
    - Instance of model class with methods `fit`, `predict_proba` and `predict`

`title` (str)
    - Figure title

----

### Example

```python
from NearestNeighbours import NearestNeighbours
from utils import visualise_synthetic

knn_model = NearestNeighbours(k=9)

visualise_synthetic(knn_model, "KNN")
```

This code produces the output:

<img src="static/knn_example.png"/>

## Algorithms Usage

Each of the algorithms is implemented using a dedicated class with the following methods:

----

**`fit(X_train, y_train)`** - Fit the model to training data.

Parameters:

`X_train` (np.ndarray)
    - Training data of shape (n_samples, m_features)

`y_train` (np.ndarray)
    - Target values of shape (n_samples,)

Returns:

`Self`
    - The fitted model

----

**`predict(X_test)`** - Predict the target variable for the test data.

Parameters:

`X_test` (np.ndarray)
    - Test data of shape (n_samples, m_features)

Returns:

`y_pred`: (np.ndarray)
    - Predicted target values of shape (n_samples,).

----
