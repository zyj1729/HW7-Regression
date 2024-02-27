# Imports
import pytest
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_prediction():
    """
    Test that the logistic regression model's predictions are within the valid probability range [0, 1].
    This ensures that the sigmoid function is applied correctly to the linear model's outputs.
    """
    X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.00001, max_iter=5000, batch_size=100)
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    pred = log_model.make_prediction(X_val)
    assert (pred >= 0).all() and (pred <= 1).all(), "Predicted value should be between 0 and 1"

def test_loss_function():
    """
    Test that the logistic regression model computes the loss function correctly.
    This ensures that the binary cross-entropy loss is calculated properly.
    """
    X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.00001, max_iter=5000, batch_size=100)
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    pred = log_model.make_prediction(X_val)
    loss = log_model.loss_function(y_val, pred)
    assert isinstance(loss, float), "Loss should be float"
    assert loss >= 0, "Loss should be at least 0"

def test_gradient():
    """
    Test that the logistic regression model computes the gradient of the loss function correctly.
    This ensures that the model can update its weights based on the gradient descent algorithm.
    """
    X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.00001, max_iter=5000, batch_size=100)
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    grad = log_model.calculate_gradient(y_val, X_val)
    assert grad.shape[0] == log_model.W.shape[0], "Gradient should have the same size as model weights"
    assert sum([isinstance(i, float) for i in grad]) == grad.shape[0], "Gradient should be float"

def test_training():
    """
    Test that the logistic regression model updates its weights after training.
    This ensures that the training process is capable of modifying the model's parameters.
    """
    X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.00001, max_iter=5000, batch_size=100)
    init_W = log_model.W
    log_model.train_model(X_train, y_train, X_val, y_val)
    out_W = log_model.W
    assert (init_W == out_W).sum() != init_W.shape[0], "Weights should be changed after training"