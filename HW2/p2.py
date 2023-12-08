import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from p2_preprocessing import preprocessing_data


def linear_train(train_data, test_data):
    model = LinearRegression()
    model.fit(train_data["X"], train_data["Y"])
    train_score = mean_squared_error(train_data["Y"], model.predict(train_data["X"]))
    test_score = mean_squared_error(test_data["Y"], model.predict(test_data["X"]))
    train_moneys = simulation(model, train_data)
    test_moneys = simulation(model, test_data)

    return (
        train_score,
        test_score,
        train_moneys,
        test_moneys,
        model.intercept_,
        model.coef_,
    )


def Quadratic_train(train_data, test_data):
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(train_data["X"], train_data["Y"])
    train_score = mean_squared_error(train_data["Y"], model.predict(train_data["X"]))
    test_score = mean_squared_error(test_data["Y"], model.predict(test_data["X"]))
    train_moneys = simulation(model, train_data)
    test_moneys = simulation(model, test_data)

    return (
        train_score,
        test_score,
        train_moneys,
        test_moneys,
    )


def simulation(model, data):
    moneys = []
    money = 1.0
    if model == None:
        for i in range(len(data["X"])):
            money *= 1 + data["Y"][i]
            moneys.append(money - 1)
        return moneys

    for i in range(len(data["X"])):
        predict = np.sign(model.predict([data["X"][i]]))
        if predict[0] > 0:
            money *= 1 + data["Y"][i]
        moneys.append(money - 1)
    return moneys


def print_data_distribution(data, intercept, coef, plot_line, title, name):
    up = {"X1": [], "X2": [], "Y": []}
    down = {"X1": [], "X2": [], "Y": []}
    for i in range(len(data["X"])):
        if data["y"][i] == 1:
            up["X1"].append(data["X"][i][0] * 100)
            up["X2"].append(data["X"][i][1] * 100)
            up["Y"].append(data["Y"][i])
        else:
            down["X1"].append(data["X"][i][0] * 100)
            down["X2"].append(data["X"][i][1] * 100)
            down["Y"].append(data["Y"][i])

    if plot_line:
        X = [data["X"][i][1] for i in range(len(data["X"]))]
        y = np.linspace(
            min(X) * 100,
            max(X) * 100,
            100,
        )
        x = (-(intercept * 100 + coef[1] * 100) * y) / (coef[0] * 100)
        plt.plot(x, y, c="k")

    plt.xlabel("GSPC")
    plt.ylabel("EURUSD")
    plt.title(title)
    plt.scatter(up["X1"], up["X2"], c="r", label="UP")
    plt.scatter(down["X1"], down["X2"], c="g", label="DOWN")
    plt.legend(loc="upper right")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.savefig("./image_p2/" + name + ".png")
    plt.clf()


if __name__ == "__main__":
    train_data, test_data = preprocessing_data()

    print_data_distribution(
        train_data,
        0,
        [0, 0],
        False,
        "Train data distribtion",
        "Train_data_distribtion",
    )
    print_data_distribution(
        test_data, 0, [0, 0], False, "Test data distribtion", "Test_data_distribtion"
    )

    if True:
        train_all_buy = simulation(None, train_data)
        test_all_buy = simulation(None, test_data)

        (
            linear_train_score,
            linear_test_score,
            linear_train_moneys,
            linear_test_moneys,
            intercept,
            coef,
        ) = linear_train(train_data, test_data)
        print("Linear Regression error on train data:", linear_train_score)
        print("Linear Regression error on test data:", linear_test_score)

        (
            quadratic_train_score,
            quadratic_test_score,
            quadratic_train_moneys,
            quadratic_test_moneys,
        ) = Quadratic_train(train_data, test_data)
        print("Quadratic Regression error on train data:", quadratic_train_score)
        print("Quadratic Regression error on test data:", quadratic_test_score)

        plt.title("Train Data Simulation")
        plt.xlabel("Day")
        plt.ylabel("Cumulative Gain/Loss(%)")
        plt.plot(train_all_buy, label="all buy", c="k")
        plt.plot(linear_train_moneys, label="linear", c="orange")
        plt.plot(quadratic_train_moneys, label="quadratic", c="b")
        plt.legend(loc="upper left")
        plt.savefig("./image_p2/moneys_train.png")
        plt.clf()

        plt.title("Test Data Simulation")
        plt.xlabel("Day")
        plt.ylabel("Cumulative Gain/Loss(%)")
        plt.plot(test_all_buy, label="all buy", c="k")
        plt.plot(linear_test_moneys, label="linear", c="orange")
        plt.plot(quadratic_test_moneys, label="quadratic", c="b")
        plt.legend(loc="upper left")
        plt.savefig("./image_p2/moneys_test.png")
        plt.clf()

        print_data_distribution(
            train_data,
            intercept,
            coef,
            True,
            "Linear Regression on Train Data",
            "Linear_Regression_on_Train_Data",
        )
        print_data_distribution(
            test_data,
            intercept,
            coef,
            True,
            "Linear Regression on Test Data",
            "Linear_Regression_on_Test_Data",
        )
