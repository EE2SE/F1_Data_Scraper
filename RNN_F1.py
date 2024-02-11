import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden


# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the input size, hidden size, and output size
input_size = 138  # Number of features in your input data
hidden_size = 64  # You can adjust this based on your problem
output_size = 1  # Output size depends on your task (regression, classification, etc.)

# Create an instance of the SimpleRNN model
model = SimpleRNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load CSV data
train_data = pd.read_csv("train.csv")
train_targets = pd.read_csv("train_targets.csv")
test_data = pd.read_csv("test.csv")
test_targets = pd.read_csv("test_targets.csv")

# convert to tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
train_target_tensor = torch.tensor(train_targets, dtype=torch.float32)
train_dataset = TensorDataset(train_tensor, train_target_tensor)

test_tensor = torch.tensor(test_data, dtype=torch.float32)
test_target_tensor = torch.tensor(test_targets, dtype=torch.float32)
test_dataset = TensorDataset(test_tensor, test_target_tensor)

# Assuming you have a DataLoader for your training data
# Adjust batch_size based on your available memory
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Number of training epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Initialize the hidden state for each epoch
    hidden = torch.zeros(1, batch_size, hidden_size)

    # Iterate over the training data
    for input_seq_batch, target_batch in train_data_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output, hidden = model(input_seq_batch, hidden)

        # Compute the loss
        loss = criterion(output, target_batch.float())  # Assuming your target is a tensor of floats

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print the loss for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the trained model for predictions
