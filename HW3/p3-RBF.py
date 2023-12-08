import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def generate_data():
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x) ** 2 + np.random.normal(0, 0.1, len(x))
    return x, y


def train_rbf(x, y, hidden_units):
    rbf_models = {}

    for m in hidden_units:
        # Select m random centroids from the input data
        centroid_indices = np.random.choice(len(x), m, replace=False)
        centroids = x[centroid_indices]

        # Create RBF model
        model = Rbf(centroids, y[centroid_indices], function="gaussian")

        rbf_models[m] = model

    return rbf_models


def plot_rbf(mlp_models):
    x_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    predictions = {m: model(x_test) for m, model in mlp_models.items()}

    plt.figure(figsize=(12, 8))
    for m, pred in predictions.items():
        plt.plot(x_test, pred, label=f"m={m}")
        plt.scatter(x, y, color="red", label="Training Data")
        plt.plot(
            x_test,
            np.sin(2 * np.pi * x_test) ** 2,
            color="black",
            label="True Function",
            linestyle="--",
        )
        plt.title("RBF Regression with Different Hidden Units")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(f"./image/RBF/p3-RBF-{m}.png")
        plt.clf()


if __name__ == "__main__":
    hidden_units = [2, 5, 10, 15, 20, 25]
    x, y = generate_data()
    mlp_models = train_rbf(x, y, hidden_units)
    plot_rbf(mlp_models)
