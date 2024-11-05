import numpy as np

# Assuming you've already loaded the MNIST dataset
# Example arrays: x_train, x_test, y_train, y_test

# Load the MNIST data (assuming the npz file format)
loaded = np.load('workspaces/MNIST_workspace/data/mnist.npz')

# Replace 'data' with the correct key
data = loaded['x_train']  # Example key


# Combine training and test datasets
x_train, y_train = loaded['x_train'], loaded['y_train']
x_test, y_test = loaded['x_test'], loaded['y_test']

data = np.concatenate((x_train, x_test), axis=0)  # Combine features
names = np.concatenate((y_train, y_test), axis=0)  # Combine labels


# Output the shapes to verify
print(f'Data shape: {data.shape}')  # Should be (70000, 28, 28) for MNIST
print(f'Names shape: {names.shape}')  # Should be (70000,)

np.savez('workspaces/MNIST_workspace/data/combined_mnist.npz', data=data, names=names)



