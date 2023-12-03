import torch
import torch.nn as nn
import torch.optim as optim

# Definition of a simple neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Set input, hidden, and output dimensions
input_size = 5
hidden_size = 10
output_size = 2

# Create an instance of the neural network
neural_network = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# Define the criterion (loss function) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neural_network.parameters(), lr=0.01)

# Example input data
input_data = torch.randn(3, input_size)

# Example labels
labels = torch.tensor([0, 1, 0])

# Train the neural network
epochs = 1000
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = neural_network(input_data)

    # Calculate the loss
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print the loss value every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Example usage of the trained neural network for prediction
test_input = torch.randn(1, input_size)
predicted_output = neural_network(test_input)
print("Predicted Output:", predicted_output.argmax(dim=1))
