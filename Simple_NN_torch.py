import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
hidden_size = 10024
learning_rate = 100
batch_size = 64
num_epochs = 100

class MyNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNeuralNet,self).__init__()
        self.hidden1_pre = nn.Linear(input_size,hidden_size)
        self.hidden1_post = nn.ReLU()
        self.hidden2_pre = nn.Linear(hidden_size,hidden_size)
        self.hidden2_post = nn.ReLU()
        self.output_pre = nn.Linear(hidden_size,output_size)
        self.output_post = nn.Softmax()

    def forward(self,vals):
        out = self.hidden1_pre(vals)
        out = self.hidden1_post(out)
        out = self.hidden2_pre(out)
        out = self.hidden2_post(out)
        out = self.output_pre(out)
        out = self.output_post(out)

        return out



def encodeClassLabels(targets: np.ndarray) -> np.ndarray:
    one_hot = np.eye(targets.max() + 1)[targets]
    return one_hot.squeeze()

if __name__ == "__main__":

    train_data = pd.read_csv("train.csv")
    train_data = train_data.drop('Unnamed: 0', axis=1)
    train_targets = pd.read_csv("train_targets.csv")
    train_targets = train_targets - 1
    train_targets = train_targets.drop('Unnamed: 0', axis=1)
    train_targets = np.array(train_targets)
    train_data = np.array(train_data)
    train_data = train_data[(~(train_targets == 24)).squeeze(),:]
    train_data: Tensor = torch.tensor(train_data,dtype= torch.float32)
    train_targets = train_targets[(~(train_targets == 24)).squeeze(),:]
    train_targets_onehot = encodeClassLabels(train_targets)
    train_targets_onehot: Tensor = torch.tensor(train_targets_onehot,dtype= torch.float32)

    test_data = pd.read_csv("test.csv")
    test_data = test_data.drop('Unnamed: 0', axis=1)
    test_targets = pd.read_csv("test_targets.csv")
    test_targets = test_targets - 1
    test_targets = test_targets.drop('Unnamed: 0', axis=1)
    test_targets = np.array(test_targets)
    test_data = np.array(test_data)
    test_data = test_data[(~(test_targets == 24)).squeeze(), :]
    test_data: Tensor = torch.tensor(test_data, dtype=torch.float32)
    test_targets = test_targets[(~(test_targets == 24)).squeeze(), :]
    test_targets_onehot = encodeClassLabels(test_targets)
    test_targets_onehot: Tensor = torch.tensor(test_targets_onehot, dtype=torch.float32)
    # model initialisation
    model = MyNeuralNet(train_data.shape[1],hidden_size,train_targets_onehot.shape[1]).to(device)

    # loss and optimiser
    criterion = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(),lr=learning_rate)

    # dataloader
    loader = DataLoader(list(zip(train_data,train_targets_onehot)), shuffle=True, batch_size = batch_size)
    test_loader = DataLoader(list(zip(test_data, test_targets_onehot)), shuffle=False, batch_size=batch_size)
    total_steps = len(loader)

    tot_loss = list()
    for i in range(num_epochs):
        for j, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # forward
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # back and optim
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            tot_loss.append(loss.item())
            if (j+1) % 10 == 0:
                print(f'Epoch [{i+1}/{num_epochs}], Step: [{j+1}/{total_steps}], Loss: {loss.item():.4f}')


    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = model(x_test)
            predicted = torch.argmax(outputs,1)

            n_correct += (torch.argmax(y_test,1) == torch.argmax(outputs,1)).sum().item()

        print(f'Accuracy: {n_correct/n_samples}')

    plt.figure()
    plt.plot(tot_loss)