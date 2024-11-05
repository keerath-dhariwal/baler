
import numpy as np

# Load the .npz file
Data = np.load('workspaces/MNIST_workspace/data/mnist.npz')

# Print the keys in the .npz file
print(Data.files)
