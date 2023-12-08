import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def generate_pure_data(size=30, normal_distribution=1):
    x = np.random.rand(size)
    y = 0.2 * np.random.normal(0, normal_distribution, size)
    return x, y


def generate_sin_data(size=30, normal_distribution=1):
    x = np.random.rand(size)
    y = np.sin(2 * np.pi * x) ** 2 + 0.2 * np.random.normal(
        0, normal_distribution, size
    )
    return x, y


def transform(xs, m):
    sin_cos_x = []
    for x in xs:
        sin_x = [np.sin(x * j) for j in range(1, m + 1)]
        cos_x = [np.cos(x * j) for j in range(1, m + 1)]
        sin_cos_x.append(sin_x + cos_x)
    return sin_cos_x


def fpe(degree, size, Remp):
    p = degree / size
    return Remp * ((1 + p) / (1 - p))


def gcv(degree, size, Remp):
    p = degree / size
    return Remp * ((1 - p) ** -2)


def vc(degree, size, Remp):
    p = degree / size
    r = 1 - np.sqrt(p - p * np.log(p) + np.log(size) / (2 * size))
    if r < 0:
        return 0
    return Remp / r


def train(train_x, train_y, test_x, test_y, max_degree, eval_func=None):
    size = len(train_x)
    eval_results = []
    for degree in range(1, max_degree + 1):
        model = LinearRegression()
        train_trans_x = transform(train_x, degree)

        eval_result = 0
        if eval_func == None:
            result = cross_val_score(
                model, train_trans_x, train_y, cv=size, scoring="neg_mean_squared_error"
            )
            eval_result = abs(result.mean())
        else:
            model.fit(train_trans_x, train_y)
            mse = mean_squared_error(train_y, model.predict(train_trans_x))
            eval_result = eval_func(degree * 2 + 1, size, abs(mse))
        eval_results.append(eval_result)

    best_degree = np.argmin(eval_results) + 1
    model = LinearRegression()
    train_trans_x = transform(train_x, best_degree)
    test_trans_x = transform(test_x, best_degree)
    model.fit(train_trans_x, train_y)
    best_degree_mse = mean_squared_error(test_y, model.predict(test_trans_x))

    return best_degree, best_degree_mse


def trainer(size, noise, epochs, max_degree, generate_data=generate_pure_data):
    fpe_degrees, fpe_risks = [], []
    gev_degrees, gev_risks = [], []
    vc_degrees, vc_risks = [], []
    cv_degrees, cv_risks = [], []
    for i in trange(epochs):
        train_x, train_y = generate_data(size, noise)
        test_x, test_y = generate_data(size, noise)

        fpe_degree, fpe_risk = train(train_x, train_y, test_x, test_y, max_degree, fpe)
        fpe_degrees.append(fpe_degree)
        fpe_risks.append(fpe_risk)

        gev_degree, gev_risk = train(train_x, train_y, test_x, test_y, max_degree, gcv)
        gev_degrees.append(gev_degree)
        gev_risks.append(gev_risk)

        vc_degree, vc_risk = train(train_x, train_y, test_x, test_y, max_degree, vc)
        vc_degrees.append(vc_degree)
        vc_risks.append(vc_risk)

        cv_degree, cv_risk = train(train_x, train_y, test_x, test_y, max_degree, None)
        cv_degrees.append(cv_degree)
        cv_risks.append(cv_risk)

    best_degrees = [fpe_degrees, gev_degrees, vc_degrees, cv_degrees]
    best_risks = [fpe_risks, gev_risks, vc_risks, cv_risks]
    return best_degrees, best_risks


def print_data_distribution(size, normal_distribution, func, name):
    x, y = func(size, normal_distribution)
    plt.scatter(x, y)
    plt.title("Data Distribution")
    if func == generate_sin_data:
        plt.plot(
            np.linspace(0, 1, 100),
            np.sin(2 * np.pi * np.linspace(0, 1, 100)) ** 2,
            c="red",
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("./image_p1/" + name + ".png")
    plt.clf()


if __name__ == "__main__":
    print_data_distribution(100, 1, generate_pure_data, "Pure1_Data_Distribution")
    print_data_distribution(100, 1, generate_sin_data, "Sin1_Data_Distribution")
    print_data_distribution(100, 0.2, generate_sin_data, "Sin0.2_Data_Distribution")

    if True:
        best_degrees, best_risks = trainer(
            size=30, noise=1, epochs=300, max_degree=6, generate_data=generate_pure_data
        )

        plt.boxplot(best_degrees, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.title("Gussian Noise Data set wtih alpha=1")
        plt.ylabel("Degree of Freedom(m)")
        plt.savefig("./image_p1/Pure1_Drgree.png")
        plt.clf()

        plt.boxplot(best_risks, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.yscale("log")
        plt.title("Gussian Noise Data set wtih alpha=1")
        plt.ylabel("Risk(MSE)")
        plt.savefig("./image_p1/Pure1_Risk(MSE).png")
        plt.clf()

    if True:
        best_degrees, best_risks = trainer(
            size=100,
            noise=0.2,
            epochs=300,
            max_degree=20,
            generate_data=generate_sin_data,
        )

        plt.boxplot(best_degrees, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.title("Sine squared func with Noise(alpha=0.2) Data set")
        plt.ylabel("Degree of Freedom(m)")
        plt.savefig("./image_p1/Sin0.2_Drgree.png")
        plt.clf()

        plt.boxplot(best_risks, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.yscale("log")
        plt.title("Sine squared func with Noise(alpha=0.2) Data set")
        plt.ylabel("Risk(MSE)")
        plt.savefig("./image_p1/Sin0.2_Risk(MSE).png")
        plt.clf()

    if True:
        best_degrees, best_risks = trainer(
            size=100,
            noise=1,
            epochs=300,
            max_degree=20,
            generate_data=generate_sin_data,
        )

        plt.boxplot(best_degrees, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.title("Sine squared func with Noise(alpha=1) Data set")
        plt.ylabel("Degree of Freedom(m)")
        plt.savefig("./image_p1/Sin1_Drgree.png")
        plt.clf()

        plt.boxplot(best_risks, labels=["fpe", "gev", "vc", "cv"], sym="")
        plt.yscale("log")
        plt.title("Sine squared func with Noise(alpha=1) Data set")
        plt.ylabel("Risk(MSE)")
        plt.savefig("./image_p1/Sin1_Risk(MSE).png")
        plt.clf()
