
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset from the .npz file
data = np.load('workspaces/MNIST_workspace/data/mnist.npz')
x_train, y_train = data['x_train'], data['y_train']

# Display a grid of the first 9 images in the training dataset
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
