
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset from the .npz file
data = np.load('workspaces/MNIST_workspace/data/mnist.npz')
x_train, y_train = data['x_train'], data['y_train']

# Display a sample image from the dataset
plt.figure(figsize=(5, 5))
plt.imshow(x_train[3], cmap='gray')  # Show the [nth] image in the training set
plt.title(f'Label: {y_train[3]}')  # Show the [nth] label
plt.axis('off')  # Hide axis
plt.show()
