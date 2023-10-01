import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


class Trigonometric_model(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.linear = torch.nn.Linear(k, 1)

    def forward(self, x):
        list_x = []
        for i in range(1, self.k + 1):
            list_x.append(torch.cos(2 * np.pi * i * x))
        tensor_x = torch.tensor(list_x, dtype=torch.float32)
        return self.linear(tensor_x)


class Algebraic_model(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.linear = torch.nn.Linear(k, 1)

    def forward(self, x):
        list_x = []
        for i in range(1, self.k + 1):
            list_x.append(x**i)
        tensor_x = torch.tensor(list_x, dtype=torch.float32)
        return self.linear(tensor_x)


def generate_data(size=10):
    x = np.random.uniform(0, 1, size=size)
    y = x**2 + 0.1 * x + np.random.normal(0, 0.5, size=size)
    x = x.tolist()
    y = y.tolist()
    for i in range(size):
        x[i] = torch.tensor([x[i]])
        y[i] = torch.tensor([y[i]])
    return x, y


def train_trigonometric_model(weight_size, epoch_size, x, y):
    data_set_size = len(x)
    trigonometric_model = Trigonometric_model(weight_size)
    optimizer = torch.optim.Adam(trigonometric_model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    loss_path = []
    for epoch in tqdm.trange(epoch_size):
        optimizer.zero_grad()
        epoch_loss = 0
        for i in range(data_set_size):
            pred = trigonometric_model(x[i])
            loss = loss_fn(pred, y[i])
            loss.backward()
            epoch_loss += loss.item()
        loss_path.append(epoch_loss)
        optimizer.step()
    return loss_path


def train_algebraic_model(weight_size, epoch_size, x, y):
    data_set_size = len(x)
    trigonometric_model = Algebraic_model(weight_size)
    optimizer = torch.optim.Adam(trigonometric_model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    loss_path = []
    for epoch in tqdm.trange(epoch_size):
        optimizer.zero_grad()
        epoch_loss = 0
        for i in range(data_set_size):
            pred = trigonometric_model(x[i])
            loss = loss_fn(pred, y[i])
            loss.backward()
            epoch_loss += loss.item()
        loss_path.append(epoch_loss)
        optimizer.step()
    return loss_path


def find_best_k(x, y):
    epoch_size = 100
    tri_k_loss_path = []
    alg_k_loss_path = []
    for k in range(1, 100):
        tri_loss_path = train_trigonometric_model(k, epoch_size, x, y)
        alg_loss_path = train_algebraic_model(k, epoch_size, x, y)
        plt.plot(tri_loss_path, label="trigonometric")
        plt.plot(alg_loss_path, label="algebraic")
        plt.legend()
        plt.savefig("./p3_figure/p3_" + "k=" + str(k) + ".png")
        plt.clf()
        tri_k_loss_path.append(tri_loss_path[-1])
        alg_k_loss_path.append(alg_loss_path[-1])

    plt.plot(tri_k_loss_path, label="trigonometric")
    plt.plot(alg_k_loss_path, label="algebraic")
    plt.legend()
    plt.savefig("p3.png")


if __name__ == "__main__":
    data_set_size = 10
    x, y = generate_data(data_set_size)
    find_best_k(x, y)
