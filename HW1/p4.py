import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


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


def train_trigonometric_model(
    weight_size, epoch_size, train_x, train_y, test_x, test_y
):
    train_date_size = len(train_x)
    test_data_size = len(test_x)
    trigonometric_model = Trigonometric_model(weight_size)
    optimizer = torch.optim.Adam(
        trigonometric_model.parameters(), lr=0.1, weight_decay=0.01
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epoch_size):
        optimizer.zero_grad()
        for i in range(train_date_size):
            pred = trigonometric_model(train_x[i])
            loss = loss_fn(pred, train_y[i])
            loss.backward()
        optimizer.step()

    loss = []
    with torch.no_grad():
        for i in range(test_data_size):
            loss.append(loss_fn(trigonometric_model(test_x[i]), test_y[i]).item())

    return np.mean(loss)


def train_algebraic_model(weight_size, epoch_size, train_x, train_y, test_x, test_y):
    train_date_size = len(train_x)
    test_data_size = len(test_x)
    algebraic_model = Algebraic_model(weight_size)
    optimizer = torch.optim.Adam(
        algebraic_model.parameters(), lr=0.1, weight_decay=0.01
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epoch_size):
        optimizer.zero_grad()
        for i in range(train_date_size):
            pred = algebraic_model(train_x[i])
            loss = loss_fn(pred, train_y[i])
            loss.backward()
        optimizer.step()

    loss = []
    with torch.no_grad():
        for i in range(test_data_size):
            loss.append(loss_fn(algebraic_model(test_x[i]), test_y[i]).item())

    return np.mean(loss)


def find_best_k(x, y):
    epoch_size = 20
    k_size = 50
    tri_k_loss_path = []
    alg_k_loss_path = []
    kfold = KFold(n_splits=5, shuffle=True)
    for k in tqdm.trange(1, k_size + 1):
        tri_test_loss = []
        alg_test_loss = []
        for train, test in kfold.split(x):
            train_x, train_y = [x[i] for i in train], [y[i] for i in train]
            test_x, test_y = [x[i] for i in test], [y[i] for i in test]
            tri_test_loss.append(
                train_trigonometric_model(
                    k, epoch_size, train_x, train_y, test_x, test_y
                )
            )
            alg_test_loss.append(
                train_algebraic_model(k, epoch_size, train_x, train_y, test_x, test_y)
            )
        tri_k_loss_path.append(np.mean(tri_test_loss))
        alg_k_loss_path.append(np.mean(alg_test_loss))

    plt.plot(tri_k_loss_path, label="trigonometric")
    plt.plot(alg_k_loss_path, label="algebraic")
    plt.legend()
    plt.savefig("p4.png")

    tri_min_index = np.argmin(tri_k_loss_path)
    print(
        "trigonometric min k:",
        tri_min_index + 1,
        "value: ",
        tri_k_loss_path[tri_min_index],
    )
    alg_min_index = np.argmin(alg_k_loss_path)
    print(
        "algebraic min k:",
        alg_min_index + 1,
        "value:",
        alg_k_loss_path[alg_min_index],
    )


if __name__ == "__main__":
    data_set_size = 100
    x, y = generate_data(data_set_size)

    find_best_k(x, y)
