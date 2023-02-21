import numpy as np
from typing import Tuple
import pandas as pd


# Define the sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Define the sine function
def sine(x):
    return np.sin(x)


def fit_sin_function(hidden_size: int,
                     learning_rate: float,
                     num_epochs: int,
                     batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Generate some sample data
    x_train = np.random.uniform(0, 2 * np.pi, size=(1000, 1))
    y_train = sine(x_train)

    # Define the neural network architecture
    input_size = 1
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))

    # Train the neural network
    losses = []
    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Split the training data into batches
        num_batches = x_train.shape[0] // batch_size
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            # Forward pass
            hidden = sigmoid(np.dot(x_batch, W1) + b1)
            y_pred = np.dot(hidden, W2) + b2

            # Compute the loss
            loss = np.square(y_pred - y_batch).sum()

            # Backward pass
            grad_y_pred = 2.0 * (y_pred - y_batch)
            grad_W2 = np.dot(hidden.T, grad_y_pred)
            grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)
            grad_hidden = np.dot(grad_y_pred, W2.T) * hidden * (1 - hidden)
            grad_W1 = np.dot(x_batch.T, grad_hidden)
            grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

            # Update the parameters
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

        # Track the loss
        losses.append(loss)

        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            print("Epoch {0}: loss = {1}".format(epoch, loss))

    # Evaluate the model on some test data
    x_test = np.linspace(0, 2 * np.pi, num=100)
    y_test = sine(x_test)
    hidden = sigmoid(np.dot(x_test.reshape(-1, 1), W1) + b1)
    y_pred = np.dot(hidden, W2) + b2

    y_pred_plot = y_pred.reshape(-1,)
    d = {'x_test': x_test, 'y_test': y_test, 'y_pred': y_pred_plot}
    df_plot = pd.DataFrame(d)

    d = {'epoch': [epoch for epoch in range(0, num_epochs)], 'loss': losses}
    df_loss = pd.DataFrame(d)

    return df_plot, df_loss

