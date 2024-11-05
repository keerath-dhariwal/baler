
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


# Call the function and load the data
(x_train, y_train), (x_test, y_test) = load_data(
    'workspaces/MNIST_workspace/data/mnist.npz')

# Print shapes to verify
print(
    f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")


# Example data for training and test accuracy and loss
epochs = np.arange(1, 21)  # 20 epochs
train_loss = np.random.rand(20) * 0.2 + 0.1  # Simulated training loss
test_loss = np.random.rand(20) * 0.2 + 0.2  # Simulated test loss
train_acc = np.random.rand(20) * 0.1 + 0.8  # Simulated training accuracy
test_acc = np.random.rand(20) * 0.1 + 0.7  # Simulated test accuracy

# Plotting loss
plt.figure(figsize=(10, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, test_loss, label='Test Loss', marker='o')
plt.title('Training vs Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, test_acc, label='Test Accuracy', marker='o')
plt.title('Training vs Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
