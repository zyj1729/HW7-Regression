![BuildStatus](https://github.com/zyj1729/HW7-Regression/actions/workflows/main.yml/badge.svg?event=push)
# Logistic Regression Model

## Method Descriptions

### `make_prediction(self, X) -> np.array`
Computes the probability estimates for the given input features using the logistic function.

**Parameters:**
- `X (np.ndarray)`: Input feature matrix where each row represents a sample.

**Returns:**
- `np.array`: The predicted probabilities that each input sample belongs to the positive class.

### `loss_function(self, y_true, y_pred) -> float`
Calculates the binary cross-entropy loss between true labels and predicted probabilities.

**Parameters:**
- `y_true (np.array)`: True binary labels for each input sample.
- `y_pred (np.array)`: Predicted probabilities for each input sample being in the positive class.

**Returns:**
- `float`: The mean binary cross-entropy loss over all input samples.

### `calculate_gradient(self, y_true, X) -> np.ndarray`
Computes the gradient of the binary cross-entropy loss function with respect to the model weights.

**Parameters:**
- `y_true (np.array)`: True binary labels for each input sample.
- `X (np.ndarray)`: Input feature matrix where each row represents a sample.

**Returns:**
- `np.ndarray`: The gradient of the loss with respect to the weights, used for updating the weights.

### `train_model(self, X_train, y_train, X_val, y_val)`
Trains the logistic regression model using the given training data.

**Parameters:**
- `X_train (np.ndarray)`: The feature matrix for the training data.
- `y_train (np.array)`: The labels for the training data.
- `X_val (np.ndarray)`: The feature matrix for the validation data.
- `y_val (np.array)`: The labels for the validation data.

Trains the model and updates the weights based on the specified learning rate and other hyperparameters.

