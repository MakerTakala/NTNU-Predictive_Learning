import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def generate_data(size, dis):
    x = np.random.rand(size)
    y = np.sin(2 * np.pi * x) ** 2 + 0.2 * np.random.normal(0, dis, size)
    return x, y


# def rbf(x, alpha):
#     return np.exp(-((x**2) / (2 * alpha**2)))


# def transformer(xs, m):
#     return [[rbf(x, 1)] * m for x in xs]


# def train_model(x, y, m):
#     model = LinearRegression()
#     model.coef_ = np.random.rand(m)
#     print(model.coef_)
#     trans_x = transformer(x, m)
#     model.fit(trans_x, y)
#     print(model.coef_)

#     line_x = np.linspace(0, 1, 1000)
#     trans_line_x = transformer(line_x, m)
#     line_y = model.predict(trans_line_x)
#     plt.plot(line_x, line_y, "r-")
#     plt.scatter(x, y)
#     plt.savefig(f"./image/p3_{m}.png")
#     plt.clf()


def train_MLP_model(x, y, m):
    model = MLPRegressor(
        hidden_layer_sizes=(
            m,
            m,
            m,
            m,
        ),
        activation="relu",
        solver="lbfgs",
        max_iter=10000000,
    )
    # print(x.reshape(-1, 1))
    model.fit(x.reshape(-1, 1), y)
    # print(model.coefs_, model.n_layers_)

    line_x = np.linspace(0, 1, 1000).reshape(-1, 1)
    line_y = model.predict(line_x)
    # print(line_y)
    plt.plot(line_x, line_y, "r-")
    plt.scatter(x, y)
    plt.savefig(f"./image/p3_{m}.png")
    plt.clf()


if __name__ == "__main__":
    x, y = generate_data(30, 0.1)

    complex_m = [2, 5, 10, 20, 25, 100]

    for m in complex_m:
        train_MLP_model(x, y, m)
