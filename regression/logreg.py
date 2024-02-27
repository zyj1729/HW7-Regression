import numpy as np
import matplotlib.pyplot as plt

# Base class for generic regressor
# (this is already complete!)
class BaseRegressor():

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):

        # Weights are randomly initialized
        self.W = np.random.randn(num_feats + 1).flatten()

        # Store hyperparameters
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats

        # Define empty lists to store losses over training
        self.loss_hist_train = []
        self.loss_hist_val = []
    
    def make_prediction(self, X):
        raise NotImplementedError
    
    def loss_function(self, y_true, y_pred):
        raise NotImplementedError
        
    def calculate_gradient(self, y_true, X):
        raise NotImplementedError
    
    def train_model(self, X_train, y_train, X_val, y_val):

        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
        # Defining intitial values for while loop
        prev_update_size = 1
        iteration = 1

        # Repeat until convergence or maximum iterations reached
        while prev_update_size > self.tol and iteration < self.max_iter:

            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()

            # Create batches
            num_batches = int(X_train.shape[0] / self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            # Create list to save the parameter update sizes for each batch
            update_sizes = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):

                # Make prediction and calculate loss
                y_pred = self.make_prediction(X_train)
                train_loss = self.loss_function(y_train, y_pred)
                self.loss_hist_train.append(train_loss)

                # Update weights
                prev_W = self.W
                grad = self.calculate_gradient(y_train, X_train)
                new_W = prev_W - self.lr * grad 
                self.W = new_W

                # Save parameter update size
                update_sizes.append(np.abs(new_W - prev_W))

                # Compute validation loss
                val_loss = self.loss_function(y_val, self.make_prediction(X_val))
                self.loss_hist_val.append(val_loss)

            # Define step size as the average parameter update over the past epoch
            prev_update_size = np.mean(np.array(update_sizes))

            # Update iteration
            iteration += 1
    
    def plot_loss_history(self):

        # Make sure training has been run
        assert len(self.loss_hist_train) > 0, "Need to run training before plotting loss history."

        # Create plot
        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(self.loss_hist_train)), self.loss_hist_train)
        axs[0].set_title('Training')
        axs[1].plot(np.arange(len(self.loss_hist_val)), self.loss_hist_val)
        axs[1].set_title('Validation')
        plt.xlabel('Steps')
        fig.tight_layout()
        plt.show()

    def reset_model(self):
        self.W = np.random.randn(self.num_feats + 1).flatten()
        self.loss_hist_train = []
        self.loss_hist_val = []
        
# Implement logistic regression as a subclass
class LogisticRegressor(BaseRegressor):
    """
    A logistic regression model that inherits from a BaseRegressor class.

    Attributes:
        num_feats (int): Number of features in the input dataset.
        learning_rate (float): The step size at each iteration while moving toward a minimum of the loss function.
        tol (float): The tolerance for stopping criteria. Training stops if the update size is less than this value.
        max_iter (int): The maximum number of iterations allowed for the training process.
        batch_size (int): The number of samples per batch to be passed through the algorithm.
        W (np.ndarray): The weights vector for the logistic regression model, including the bias term as the last entry.
    """

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):
        super().__init__(
            num_feats,
            learning_rate=learning_rate,
            tol=tol,
            max_iter=max_iter,
            batch_size=batch_size
        )
    
    def make_prediction(self, X) -> np.array:
        """
        Computes the probability estimates for the given input features using the logistic function.

        Parameters:
            X (np.ndarray): Input feature matrix where each row represents a sample.

        Returns:
            np.array: The predicted probabilities that each input sample belongs to the positive class.
        """
        # Linear combination of input features and weights
        temp = np.dot(X, self.W)
        # Logistic function applied to linear combination for probability estimation
        y_pred = 1 / (1 + np.exp(-temp))
        return y_pred
        
    
    def loss_function(self, y_true, y_pred) -> float:
        """
        Calculates the binary cross-entropy loss between true labels and predicted probabilities.

        Parameters:
            y_true (np.array): True binary labels for each input sample.
            y_pred (np.array): Predicted probabilities for each input sample being in the positive class.

        Returns:
            float: The mean binary cross-entropy loss over all input samples.
        """
        # Binary cross-entropy loss computation
        losses = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(losses)
        
        
        
    def calculate_gradient(self, y_true, X) -> np.ndarray:
        """
        Computes the gradient of the binary cross-entropy loss function with respect to the model weights.

        Parameters:
            y_true (np.array): True binary labels for each input sample.
            X (np.ndarray): Input feature matrix where each row represents a sample.

        Returns:
            np.ndarray: The gradient of the loss with respect to the weights, used for updating the weights.
        """
        # Predicted probabilities for the given input
        y_pred = self.make_prediction(X)
        # Gradient of the binary cross-entropy loss with respect to the weights
        return np.dot(X.T, y_pred - y_true) / X.shape[0]