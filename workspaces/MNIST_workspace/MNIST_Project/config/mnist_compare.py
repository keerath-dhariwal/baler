from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data = np.load('workspaces/MNIST_workspace/data/mnist.npz')
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for inference
    x_reconstructed = model(x_test)

# Assuming you already have the autoencoder model and data loaded
# Example data (x_test) and its corresponding reconstructed images (x_reconstructed)
# Replace 'model.predict(x_test)' with your model's reconstruction process


def display_reconstruction(x_test, x_reconstructed, n=10):
    """Displays original and reconstructed images side by side for comparison."""

    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_reconstructed[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage with test data and reconstructed data
# Assuming x_test contains the original images, and model.predict(x_test) gives reconstructed images
x_test = data['x_test'][:10]  # Sample 10 test images
# Replace with your model's reconstruction
x_reconstructed = model.predict(x_test)

# Display the original vs. reconstructed images
display_reconstruction(x_test, x_reconstructed)


# Compute MSE between the original and reconstructed images
mse_values = [mean_squared_error(
    x_test[i].flatten(), x_reconstructed[i].flatten()) for i in range(len(x_test))]

# Plotting MSE for each reconstructed image
plt.plot(mse_values, marker='o')
plt.title('Reconstruction Error (MSE) for Each Image')
plt.xlabel('Image Index')
plt.ylabel('MSE')
plt.show()
