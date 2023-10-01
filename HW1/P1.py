import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def pre_processsing(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)
    train_data = {
        "data": np.array(
            train_data[1].map(lambda x: float(x.strip("%")) / 100)
        ).reshape(-1, 1),
        "ans": np.array(train_data[2].map(lambda x: 0 if x == "R" else 1)),
    }
    test_data = {
        "data": np.array(test_data[1].map(lambda x: float(x.strip("%")) / 100)).reshape(
            -1, 1
        ),
        "ans": np.array(test_data[2].map(lambda x: 0 if x == "R" else 1)),
    }
    return train_data, test_data


class Knn:
    def __init__(self):
        self.train_data = None
        self.k = None

    def train_with_leave_one_out(self, train_data):
        self.train_data = train_data
        size = train_data["data"].size
        best_k = 0
        best_average = 0
        for k in range(1, size):
            knnModel = KNeighborsClassifier(n_neighbors=k)
            kfold = KFold(n_splits=size, shuffle=True)
            average = 0
            for trainset, testset in kfold.split(train_data["data"]):
                knnModel.fit(
                    np.array([train_data["data"][i] for i in trainset]).reshape(-1, 1),
                    np.array([train_data["ans"][i] for i in trainset]),
                )
                average += knnModel.score(
                    np.array([train_data["data"][i] for i in testset]).reshape(-1, 1),
                    np.array([train_data["ans"][i] for i in testset]),
                )
            average /= size
            if average > best_average:
                best_average = average
                best_k = k
        self.k = best_k

    def predict(self, test_data):
        if self.train_data is None:
            raise Exception("Need to train first")
        knnModel = KNeighborsClassifier(n_neighbors=self.k)
        knnModel.fit(self.train_data["data"], self.train_data["ans"])
        return knnModel.score(test_data["data"], test_data["ans"])


if __name__ == "__main__":
    train_data, test_data = pre_processsing(
        "./obesity_election_2004.csv", "./obesity_election_2000.csv"
    )
    knnModel = Knn()
    knnModel.train_with_leave_one_out(train_data)
    print("k: ", knnModel.k)
    train_err = 1 - knnModel.predict(train_data)
    print("train err: ", train_err)
    test_err = 1 - knnModel.predict(test_data)
    print("test err: ", 1 - test_err)
