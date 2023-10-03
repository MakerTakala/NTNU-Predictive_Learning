import numpy as np
from sklearn.linear_model import LinearRegression


def generate_data(size=10):
    xs = np.random.uniform(0, 1, size=size)
    ys = xs**2 + 0.1 * xs + np.random.normal(0, 0.5, size=size)
    return xs, ys


def cos_x_k(x, k):
    cos_x = np.array([])
    for i in range(1, k + 1):
        cos_x = np.append(cos_x, np.cos(2 * np.pi * x * i))
    return cos_x.reshape(-1, k)


if __name__ == "__main__":
    data_set_size = 10
    xs, ys = generate_data(data_set_size)

    mdoel = LinearRegression()
    mdoel.fit(cos_x_k(xs, 10), ys)
