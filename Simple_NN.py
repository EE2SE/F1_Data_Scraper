import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleNN():

    def __init__(self, input_size: int, hidden_size: int, output_size: int, data_len: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.data_count = data_len

        np.random.seed(100)
        self.i2h_w = np.random.randn(self.hidden_size, self.input_size)
        self.i2h_b = np.random.randn(self.hidden_size)

        self.h2o_w = np.random.randn(self.output_size, self.hidden_size)
        self.h2o_b = np.random.randn(self.output_size)

        self.hidden_layer_pre_act = np.zeros((self.hidden_size, data_len))
        self.hidden_layer = np.zeros((self.hidden_size, data_len))
        self.output_layer_pre_act = np.zeros((output_size, data_len))
        self.output_layer = np.zeros((output_size, data_len))

        self.dZ1 = np.zeros((hidden_size, data_len))
        self.dZ2 = np.zeros((output_size, data_len))
        self.dW2 = np.zeros((hidden_size, input_size))
        self.dW2 = np.zeros((output_size, hidden_size))
        self.db1 = np.zeros((hidden_size, 1))
        self.db2 = np.zeros((output_size, 1))



    def forward(self, input_data: np.ndarray):
        self.hidden_layer_pre_act = self.i2h_w.dot(input_data) + self.i2h_b[:,np.newaxis]
        self.hidden_layer = self.ReLU(self.hidden_layer_pre_act)
        self.output_layer_pre_act = self.h2o_w.dot(self.hidden_layer) + self.h2o_b[:,np.newaxis]
        self.output_layer = self.softmax(self.output_layer_pre_act)

    def backpropagate(self, targets: np.ndarray, input_data: np.ndarray):
        self.dZ2 = self.output_layer - targets
        self.dW2 = 1 / self.data_count * self.dZ2.dot(self.hidden_layer.T)
        self.db2 = 1 / self.data_count * np.sum(self.dZ2, 1)
        self.dZ1 = self.h2o_w.T.dot(self.dZ2) * self.der_ReLU(self.hidden_layer_pre_act)
        self.dW1 = 1 / self.data_count * self.dZ1.dot(input_data.T)
        # print(self.db1.shape)
        self.db1 = 1 / self.data_count * np.sum(self.dZ1, 1)
        # self.db1.reshape(self.hidden_size, 1)
        # print(self.db1.shape)

    def update_params(self):
        lr = 0.1

        self.h2o_w = self.h2o_w - lr * self.dW2
        self.h2o_b = self.h2o_b - lr * self.db2
        self.i2h_w = self.i2h_w - lr * self.dW1
        self.i2h_b = self.i2h_b - lr * self.db1

    def der_ReLU(self, matrix: np.ndarray):
        return matrix > 0

    def ReLU(self, matrix: np.ndarray):
        return np.maximum(0, matrix)

    def softmax(self, matrix: np.ndarray):
        return np.exp(matrix) / np.sum(np.exp(matrix), axis=0)

    def get_pred(self):
        self.predictions = np.argmax(self.output_layer, 0)
        return self.predictions

    def get_acc(self, pred, targets):
        pred = pred[:,np.newaxis]
        return np.sum(pred == targets) / len(targets)

    def test(self, input: np.ndarray, labels: np.ndarray):
        self.forward(input)
        acc = self.get_acc(self.get_pred(), labels)
        print("Test Accuracy: ", acc)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
        targets_oh = encodeClassLabels(targets)
        targets_oh = targets_oh.T
        self.acc = np.zeros((epochs, 1))

        for i in range(epochs):
            self.forward(inputs)
            self.backpropagate(targets_oh, inputs)
            self.update_params()
            self.acc[i] = self.get_acc(self.get_pred(),targets)
            if i % 100 == 0:
                print("Iteration: ", i)
                print("Accuracy", self.get_acc(self.get_pred(),targets))


def encodeClassLabels(targets: np.ndarray) -> np.ndarray:
    one_hot = np.eye(targets.max() + 1)[targets]
    return one_hot.squeeze()


if __name__ == '__main__':
    train_data = pd.read_csv("train.csv")
    train_data = train_data.drop('Unnamed: 0', axis=1)
    train_targets = pd.read_csv("train_targets.csv")
    train_targets = train_targets - 1
    train_targets = train_targets.drop('Unnamed: 0', axis=1)
    train_targets = np.array(train_targets)
    train_data = np.array(train_data)
    train_data = train_data[(~(train_targets == 24)).squeeze(),:]
    train_targets = train_targets[(~(train_targets == 24)).squeeze(),:]
    train_targets_onehot = encodeClassLabels(train_targets)

    test_data = pd.read_csv("test.csv")
    test_data = test_data.drop('Unnamed: 0', axis=1)
    test_targets = pd.read_csv("test_targets.csv")
    test_targets = test_targets - 1
    test_targets = test_targets.drop('Unnamed: 0', axis=1)
    test_targets = np.array(test_targets)
    test_data = np.array(test_data)
    test_targets_onehot = encodeClassLabels(test_targets)

    net = SimpleNN(train_data.shape[1], 32, train_targets.max() + 1, train_data.shape[0])
    net.train(train_data.T, train_targets, 100000)

    net.test(test_data.T,test_targets)

    plt.figure()
    plt.plot(net.acc)
