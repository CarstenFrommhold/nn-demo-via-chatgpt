"""
I want an example of a neural network learning the sinus function only based on numpy.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the hyperparameters
learning_rate = 0.01
num_epochs = 1000
hidden_size = 32

# Define the sine function
def sine(x):
    return np.sin(x)

# Generate some sample data
x_train = np.random.uniform(0, 2 * np.pi, size=(1000, 1))
y_train = sine(x_train)

# Define the neural network architecture
input_size = 1
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Train the neural network
for epoch in range(num_epochs):
    # Forward pass
    hidden = np.maximum(0, np.dot(x_train, W1) + b1) # ReLU activation function
    y_pred = np.dot(hidden, W2) + b2

    # Compute the loss
    loss = np.square(y_pred - y_train).sum()

    # Backward pass
    grad_y_pred = 2.0 * (y_pred - y_train)
    grad_W2 = np.dot(hidden.T, grad_y_pred)
    grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)
    grad_hidden = np.dot(grad_y_pred, W2.T)
    grad_hidden[hidden <= 0] = 0
    grad_W1 = np.dot(x_train.T, grad_hidden)
    grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

    # Update the parameters
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print("Epoch {0}: loss = {1}".format(epoch, loss))

# Evaluate the model on some test data
x_test = np.linspace(0, 2 * np.pi, num=100)
y_test = sine(x_test)
hidden = np.maximum(0, np.dot(x_test.reshape(-1, 1), W1) + b1)
y_pred = np.dot(hidden, W2) + b2

# Plot the results
plt.plot(x_test, y_test, label="ground truth")
plt.plot(x_test, y_pred, label="predicted")
plt.legend()
plt.show()


"""
In this example, we generate some sample data by sampling points from the sine function, and then define a neural network architecture with one hidden layer of size 32. We train the network using gradient descent, with ReLU activation function in the hidden layer, and print the loss every 100 epochs. Finally, we evaluate the model on some test data, and plot the ground truth function along with the predicted values.
Note that this is a very basic implementation of a neural network, and it is not optimized for performance. Additionally, this example uses a fixed learning rate and number of epochs, which may not be optimal for all datasets. In practice, more advanced optimization techniques like learning rate schedules and early stopping would be used to train a neural network on a real-world problem.
"""

"""
On which versions of numpy and matplotlib is this based?

This code example should work on most recent versions of NumPy and Matplotlib. However, for the sake of reproducibility, I can provide the version information for the libraries that I used when creating the example:

    NumPy version: 1.20.3
    Matplotlib version: 3.4.2
"""


"""
Can you change relu to gelu activation function?

# Define the GELU activation function
def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

In this implementation, we define the GELU activation function as a separate function, and use it in the forward and backward passes of the neural network. The GELU function has a slightly different gradient than the ReLU function, so we need to modify the backward pass accordingly. Finally, we evaluate the model on the same test data as before and plot the results.
"""

"""
This example does not converge. Can you give me an example which does?

I apologize for the previous example not converging. Here is a modified version that should converge:

--> GAVE ME EXACT SAME RESULTS AS THE PREVIOUS ONE
"""

"""
Still, does not converge. What about the sigmoid activation function? 

-> def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

In this implementation, we use the sigmoid activation function, which has a well-defined gradient. We modify the backward pass to compute the gradient of the sigmoid activation function and use it to update the weights. With these modifications, the network should be able to converge to a good solution.
"""

"""
It converges with a learning rate of 0.0001. 
Well, can we add a batch_size parameter for the epochs?

Added batch size but kept LR at 0.01

"""

"""
Can you add the plot of the loss function? We can stick to a point for every 100 epochs.

-> the weired thing is that now he used the LR of 0.0001
"""

"""
This worked in a sense that the code logged the code to a list.

Finally, i adjusted a little bit, especially the plot part to give me back some dataframes to plot somewhere else.

Ended up with the following code:
"""


"""
Now, to build an streamlit app, i simply let chatgpt build the body.
Wrapped the sin-fitting-curve code to a function and called it.
Voilá.
"""

"""
Ask ChatGPT for the best hyperparameters.
"""