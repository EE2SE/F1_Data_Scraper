import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

torch.autograd.set_detect_anomaly(True)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the input size, hidden size, and output size
input_size = 138  # Number of features in your input data
hidden_size = 64  # You can adjust this based on your problem
output_size = 25  # Output size depends on your task (regression, classification, etc.)

# Create an instance of the SimpleRNN model
model = SimpleRNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load CSV data
train_data = pd.read_csv("train.csv")
train_data = train_data.drop('Unnamed: 0',axis=1)
train_targets = pd.read_csv("train_targets.csv")
train_targets = train_targets - 1
train_targets = train_targets.drop('Unnamed: 0',axis=1)
test_data = pd.read_csv("test.csv")
test_data = test_data.drop('Unnamed: 0',axis=1)
test_targets = pd.read_csv("test_targets.csv")
test_targets = test_targets - 1
test_targets = test_targets.drop('Unnamed: 0',axis=1)

# convert to tensors
train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
train_target_tensor = torch.tensor(train_targets.values, dtype=torch.long)
train_dataset = TensorDataset(train_tensor, train_target_tensor)

test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
test_target_tensor = torch.tensor(test_targets.values, dtype=torch.long)
test_dataset = TensorDataset(test_tensor, test_target_tensor)

# Assuming you have a DataLoader for your training data
# Adjust batch_size based on your available memory
batch_size = 1
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Number of training epochs
num_epochs = 100
i = 0
tot = len(train_targets)

# Training loop
for epoch in range(num_epochs):

    # Initialize the hidden state for each epoch
    hid = model.initHidden()

    # Iterate over the training data
    for input_seq_batch, target_batch in train_data_loader:
        i = i+1
        print(i/tot)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output, hid = model(input_seq_batch, hid)

        # Compute the loss
        loss = criterion(output.squeeze(), target_batch.squeeze())  # Assuming your target is a tensor of floats

        # Backward pass
        loss.backward(retain_graph=True)

        # Update the weights
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    # Print the loss for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the trained model for predictions