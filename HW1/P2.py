import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def pre_processsing(data_path):
    data = pd.read_csv(data_path, header=None)
    data = {
        "data": np.array(data[1].map(lambda x: float(x.strip("%")) / 100)).reshape(
            -1, 1
        ),
        "ans": np.array(data[2].map(lambda x: 0 if x == "R" else 1)),
    }
    return data


class Knn:
    def __init__(self):
        self.train_data = None
        self.k = None

    def train_with_leave_one_out(self, train_data):
        self.train_data = train_data
        size = train_data["data"].size

        errs = []
        for k in range(1, size):
            knnModel = KNeighborsClassifier(n_neighbors=k)
            kfold = KFold(n_splits=size, shuffle=True)
            err = []
            for trainset, testset in kfold.split(train_data["data"]):
                x_train = np.array([train_data["data"][i] for i in trainset]).reshape(
                    -1, 1
                )
                y_train = np.array([train_data["ans"][i] for i in trainset])
                x_test = np.array([train_data["data"][i] for i in testset]).reshape(
                    -1, 1
                )
                y_test = np.array([train_data["ans"][i] for i in testset])

                knnModel.fit(
                    x_train,
                    y_train,
                )
                prediction = knnModel.predict(x_test)

                err.append(np.mean([x != y for x, y in zip(prediction, y_test)]))
            errs.append(np.mean(err))

        self.k = np.argmin(errs) + 1

        return errs

    def predict(self, test_data):
        if self.train_data is None:
            raise Exception("Need to train first")
        knnModel = KNeighborsClassifier(n_neighbors=self.k)
        knnModel.fit(self.train_data["data"], self.train_data["ans"])
        prediction = knnModel.predict(test_data["data"])
        return np.mean([x != y for x, y in zip(prediction, test_data["ans"])])


if __name__ == "__main__":
    train_data = pre_processsing("./obesity_election_2000.csv")
    test_data = pre_processsing("./obesity_election_2004.csv")

    knnModel = Knn()
    errs = knnModel.train_with_leave_one_out(train_data)

    print("k: ", knnModel.k)
    train_err = knnModel.predict(train_data)
    print("train err: ", train_err)
    test_err = knnModel.predict(test_data)
    print("test err: ", test_err)

    plt.plot([x for x in range(1, train_data["data"].size)], errs)
    plt.legend(["train error rate"])
    plt.xlabel("k")
    plt.ylabel("error rate")
    plt.savefig("p2.png")
