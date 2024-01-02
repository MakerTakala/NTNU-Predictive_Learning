import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from skpp import ProjectionPursuitRegressor


def generate_data(n_samples):
    def f(x):
        return 10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 5 * x[4]

    X = np.random.uniform(-1.5, 1.5, (n_samples, 6))
    Y = [f(x) for x in X]

    return X, Y


def normalize_root_mean_squared_error(y_true, y_pred):
    up = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    down = np.sum(np.array(y_true) ** 2)
    return np.sqrt(up / down)


def evaluate_models(train_size, valid_size, test_size):
    train_X, train_y = generate_data(train_size)
    valid_X, valid_y = generate_data(valid_size)
    test_X, test_y = generate_data(test_size)

    c = max(
        abs(np.mean(train_y) - 3 * np.std(train_y)),
        abs(np.mean(train_y) + 3 * np.std(train_y)),
    )

    best_mse_vaild = 100000000
    best_gamma = 0
    best_epsilon = 0
    model_mse_test = 100000000
    model_nrmse_test = 100000000

    svm_results = []
    for gamma in [2**i for i in range(-5, 6)]:
        svm_result_mse = []
        for epsilon in range(0, 10, 2):
            svm = SVR(kernel="rbf", epsilon=epsilon, gamma=gamma, C=c)
            svm.fit(train_X, train_y)
            svm_result_mse.append(
                {
                    "train": mean_squared_error(train_y, svm.predict(train_X)),
                    "valid": mean_squared_error(valid_y, svm.predict(valid_X)),
                    "test_mse": mean_squared_error(test_y, svm.predict(test_X)),
                    "test_nrmse": normalize_root_mean_squared_error(
                        test_y, svm.predict(test_X)
                    ),
                    "gamma": gamma,
                    "epsilon": epsilon,
                }
            )

            if best_mse_vaild > svm_result_mse[-1]["valid"]:
                best_mse_vaild = svm_result_mse[-1]["valid"]
                best_epsilon = epsilon
                best_gamma = gamma
                model_mse_test = svm_result_mse[-1]["test_mse"]
                model_nrmse_test = svm_result_mse[-1]["test_nrmse"]

        svm_results.append(svm_result_mse)

    print(f"best mse on valid set: {best_mse_vaild}")
    print(f"best epsilon: {best_epsilon}")
    print(f"best gamma: 2^{np.log2(best_gamma)}")
    print(f"best model mse on test set: {model_mse_test}")
    print(f"best model nrmse on test set: {model_nrmse_test}")

    print()

    best_mse_vaild = 100000000
    best_r = 0
    model_mse_test = 100000000
    model_nrmse_test = 100000000

    ppr_results = []
    for r in range(1, 21):
        ppr = ProjectionPursuitRegressor(r=r)
        ppr.fit(train_X, train_y)
        ppr_results.append(
            {
                "train": mean_squared_error(train_y, ppr.predict(train_X)),
                "valid": mean_squared_error(valid_y, ppr.predict(valid_X)),
                "test_mse": mean_squared_error(test_y, ppr.predict(test_X)),
                "test_nrmse": normalize_root_mean_squared_error(
                    test_y, ppr.predict(test_X)
                ),
                "r": r,
            }
        )
        if best_mse_vaild > ppr_results[-1]["valid"]:
            best_mse_vaild = ppr_results[-1]["valid"]
            best_r = r
            model_mse_test = ppr_results[-1]["test_mse"]
            model_nrmse_test = ppr_results[-1]["test_nrmse"]

    print(f"best mse on valid set: {best_mse_vaild}")
    print(f"best r: {best_r}")
    print(f"best model mse on test set: {model_mse_test}")
    print(f"best model nrmse on test set: {model_nrmse_test}")

    return svm_results, ppr_results


def plot_svm_table(results, name, title):
    plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=30)
    plt.axis("off")

    datas = []
    y_axis = [f"2^{i}" for i in range(-5, 6)]
    for index, result in enumerate(results):
        data = [f"γ={y_axis[index]}"]
        for item in result:
            data.append(str(item["valid"]))
        datas.append(data)
    x_axis = ["", "ε=0", "ε=2", "ε=4", "ε=6", "ε=8"]
    table = plt.table(
        colLabels=x_axis,
        cellText=datas,
        loc="center",
        cellLoc="center",
        fontsize=15,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 3)
    plt.savefig(f"./image/{name}.png")


def plot_ppr_table(results, name, title):
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=30)
    plt.axis("off")

    datas = []
    y_axis = range(1, 21)
    for index, result in enumerate(results):
        datas.append([f"r={y_axis[index]}", result["valid"]])
    x_axis = ["", "mse"]
    table = plt.table(
        colLabels=x_axis,
        cellText=datas,
        loc="center",
        cellLoc="center",
        fontsize=10,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    plt.savefig(f"./image/{name}.png")


if __name__ == "__main__":
    train_size = 100
    valid_size = 100
    test_size = 800
    svm_results, ppr_results = evaluate_models(train_size, valid_size, test_size)

    plot_svm_table(svm_results, "svm_mse", "SVM MSE on validation set")
    plot_ppr_table(ppr_results, "ppr_mse", "PPR MSE on validation set")
