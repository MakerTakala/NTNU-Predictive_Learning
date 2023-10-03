import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def generate_data(size=10):
    xs = np.random.uniform(0, 1, size=size)
    ys = xs**2 + 0.1 * xs + np.random.normal(0, 0.5, size=size)
    return xs, ys


def cos_x_k(xs, k):
    size = xs.size
    cos_xs = np.array([])
    for x in xs:
        cos_x = np.array([])
        for i in range(1, k + 1):
            cos_x = np.append(cos_x, np.cos(2 * np.pi * x * i))
        cos_xs = np.append(cos_xs, cos_x)
    return cos_xs.reshape(size, k)


def mul_x_k(xs, k):
    size = xs.size
    mul_xs = np.array([])
    for x in xs:
        mul_x = np.array([])
        for i in range(1, k + 1):
            mul_x = np.append(mul_x, x**i)
        mul_xs = np.append(mul_xs, mul_x)
    return mul_xs.reshape(size, k)


def train(xs, ys, func, name):
    average = []
    kfold = KFold(n_splits=5, shuffle=True)
    for k in range(1, data_set_size - 1):
        reg = []
        for trainset, testset in kfold.split(xs, ys):
            xs_train = np.array([xs[i] for i in trainset])
            ys_train = np.array([ys[i] for i in trainset])
            xs_test = np.array([xs[i] for i in testset])
            ys_test = np.array([ys[i] for i in testset])

            model = LinearRegression()
            model.fit(func(xs_train, k), ys_train)
            prediction = model.predict(func(xs_test, k))
            loss = mean_squared_error(ys_test, prediction)
            reg.append(regression(k, data_set_size, loss))
        average.append(np.mean(reg))

    min_idx = np.argmin(average)
    print("k: ", min_idx + 1)
    print("loss: ", average[min_idx])

    plt.plot([x for x in range(1, len(average) + 1)], average)
    plt.legend(["Schwartz Criterion"])
    plt.xlabel("k")
    plt.ylabel("Estimated")
    plt.savefig("p4" + name + ".png")


def regression(dof, n, score):
    p = dof / n
    return (1 + p * ((1 - p) ** -1) * np.log(n)) * score


if __name__ == "__main__":
    data_set_size = 10
    xs, ys = generate_data(data_set_size)

    train(xs, ys, cos_x_k, "tri")
    train(xs, ys, mul_x_k, "alg")
