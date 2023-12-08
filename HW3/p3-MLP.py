import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


def generate_data():
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x) ** 2 + np.random.normal(0, 0.1, len(x))
    return x, y


def train_mlp(x, y, hidden_units):
    mlp_models = {}

    for m in hidden_units:
        mlp = MLPRegressor(
            hidden_layer_sizes=(m,),
            activation="tanh",
            solver="lbfgs",
            max_iter=10**19,
        )
        mlp.fit(x.reshape(-1, 1), y)
        mlp_models[m] = mlp

    return mlp_models


def plot_mlp(mlp_models):
    x_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    predictions = {m: model.predict(x_test) for m, model in mlp_models.items()}

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
        plt.title("MLP Regression with Different Hidden Units")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(f"./image/MLP/p3-MLP-{m}.png")
        plt.clf()


if __name__ == "__main__":
    hidden_units = [2, 5, 10, 15, 20, 25]
    x, y = generate_data()
    mlp_models = train_mlp(x, y, hidden_units)
    plot_mlp(mlp_models)
