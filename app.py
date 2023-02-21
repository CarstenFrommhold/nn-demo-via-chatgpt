import streamlit as st

# Define the default values for the batch size and learning rate
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# Define the UI elements for the batch size and learning rate
batch_size = st.sidebar.number_input("Batch size", value=DEFAULT_BATCH_SIZE)
learning_rate = st.sidebar.number_input("Learning rate", value=DEFAULT_LEARNING_RATE)

# Define a reset button that resets the values to their defaults
if st.sidebar.button("Reset"):
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE

# Display the current values of the batch size and learning rate
st.write("Batch size:", batch_size)
st.write("Learning rate:", learning_rate)
