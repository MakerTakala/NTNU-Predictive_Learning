import numpy as np
import matplotlib.pyplot as plt


def generate_data(size=10):
    xs = np.random.uniform(0, 1, size=size)
    ys = xs**2 + 0.1 * xs + np.random.normal(0, 0.5, size=size)
    return xs, ys


if __name__ == "__main__":
    xs, ys = generate_data(100)
    plt.plot(xs, ys, "o")
    plt.savefig("./data.png")
