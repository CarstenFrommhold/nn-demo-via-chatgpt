"""
Write a streamlit app where I can parametrize the following:
- Batch size
- Learning Rate
- Number of epochs
- Hidden size

Also, add a go-for-it button
"""

import streamlit as st

# Define the default values for the batch size, learning rate, num_epochs, and hidden_size
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_HIDDEN_SIZE = 32

# Define the UI elements for the batch size, learning rate, num_epochs, and hidden_size
batch_size = st.sidebar.number_input("Batch size", value=DEFAULT_BATCH_SIZE)
learning_rate = st.sidebar.number_input("Learning rate", value=DEFAULT_LEARNING_RATE, format="%.4f")
num_epochs = st.sidebar.number_input("Number of epochs", value=DEFAULT_NUM_EPOCHS)
hidden_size = st.sidebar.number_input("Hidden size", value=DEFAULT_HIDDEN_SIZE)

# Define a go button to start the app. This will be used to trigger the training.
if st.sidebar.button("Go for it!"):
    st.sidebar.write("Training...")
