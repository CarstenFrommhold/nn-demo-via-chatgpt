import streamlit as st
from nn_from_scratch import fit_sin_function
import matplotlib.pyplot as plt
import seaborn as sns

# Define the default values for the batch size, learning rate, num_epochs, and hidden_size
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_HIDDEN_SIZE = 32

# Define the UI elements for the batch size, learning rate, num_epochs, and hidden_size
hidden_size = st.sidebar.number_input("Hidden size", value=DEFAULT_HIDDEN_SIZE)
batch_size = st.sidebar.number_input("Batch size", value=DEFAULT_BATCH_SIZE)
learning_rate = st.sidebar.number_input("Learning rate", value=DEFAULT_LEARNING_RATE)
num_epochs = st.sidebar.number_input("Number of epochs", value=DEFAULT_NUM_EPOCHS)

def plot_figures_side_by_side(fig1, fig2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(fig1)
    ax2.imshow(fig2)
    return fig

# Define a go button to start the app. This will be used to trigger the training.
if st.sidebar.button("Go for it!"):
    st.write("Training...")

    df_plot, df_loss = fit_sin_function(batch_size=batch_size,
                                        learning_rate=learning_rate,
                                        num_epochs=num_epochs,
                                        hidden_size=hidden_size)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.lineplot(data=df_loss, x="epoch", y="loss", ax=ax1, label="loss")
#    st.pyplot(fig)

#    fig = plt.figure()
    sns.scatterplot(data=df_plot, x="x_test", y="y_test", ax=ax2, label="y_test")
    sns.scatterplot(data=df_plot, x="x_test", y="y_pred", ax=ax2, label="y_pred")
    st.pyplot(fig)
