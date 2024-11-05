import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset from the .npz file
data = np.load('workspaces/MNIST_workspace/data/mnist.npz')
x_train, y_train = data['x_train'], data['y_train']

# Function to display a batch of images from the dataset


def display_batch(start_idx, batch_size=100):
    plt.figure(figsize=(10, 10))

    for i in range(batch_size):
        plt.subplot(10, 10, i + 1)
        plt.imshow(x_train[start_idx + i], cmap='gray')
        plt.title(f'{y_train[start_idx + i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Display the first 100 images
display_batch(0, batch_size=100)

# Iterate through the entire dataset in batches of 100
# for start_idx in range(0, len(x_train), 100):
#   display_batch(start_idx, batch_size=100)

#x_test, y_test = data['x_test'], data['y_test']

# Display the first 100 test images
# Change `x_train` and `y_train` to `x_test` and `y_test` for test data
#display_batch(0, batch_size=100)
