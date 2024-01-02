import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC


def generate_data(n_samples):
    X = np.random.uniform(0, 1, (n_samples, 20))
    Y = np.sign(np.sum(X[:, :10], axis=1) - 5)
    return X, Y


def get_min_max_distance(X, Y, svm):
    X_pos = X[Y == 1]
    X_neg = X[Y == -1]
    distance_pos = svm.decision_function(X_pos)
    distance_neg = svm.decision_function(X_neg)
    return (
        abs(np.max(distance_pos)),
        abs(np.min(distance_pos)),
        abs(np.max(distance_neg)),
        abs(np.min(distance_neg)),
    )


def split_data_get_dis_count(X, Y, svm, min_distance_pos, max_distance_neg):
    X_pos = X[Y == 1]
    X_neg = X[Y == -1]
    distance_pos = svm.decision_function(X_pos)
    distance_neg = svm.decision_function(X_neg)
    distance_pos = distance_pos / min_distance_pos
    distance_neg = distance_neg / max_distance_neg

    distance_pos_round = list(map(lambda x: math.ceil(x), distance_pos))
    distance_neg_round = list(map(lambda x: math.floor(x), distance_neg))
    distance_pos_count = [0] * 21
    distance_neg_count = [0] * 21
    for dis in distance_pos_round:
        distance_pos_count[int(dis + 10)] += 1
    for dis in distance_neg_round:
        distance_neg_count[int(dis + 10)] += 1

    mean = max(list(distance_pos_count) + list(distance_neg_count)) / 2

    for i in range(0, 21, 1):
        if distance_pos_count[i] == 0:
            distance_pos_count[i] = np.nan
        else:
            break
    for i in range(20, -1, -1):
        if distance_neg_count[i] == 0:
            distance_neg_count[i] = np.nan
        else:
            break

    return (distance_pos, distance_neg, distance_pos_count, distance_neg_count, mean)


def evaluate_models_plot(train_size, valid_size, test_size):
    train_X, train_Y = generate_data(train_size)
    valid_X, valid_Y = generate_data(valid_size)
    test_X, test_Y = generate_data(test_size)

    best_svm_score = 0
    best_c = 1
    for c in np.logspace(-3, 3, 10):
        svm = SVC(kernel="linear", C=c)
        svm.fit(train_X, train_Y)
        score = svm.score(valid_X, valid_Y)
        if score > best_svm_score:
            best_svm_score = score
            best_c = c

    best_svm = SVC(kernel="linear", C=best_c)
    best_svm.fit(train_X, train_Y)
    print("number of support vector: ", len(svm.support_vectors_))
    print("error on test set: ", 1 - best_svm.score(test_X, test_Y))
    print

    (
        max_distance_pos,
        min_distance_pos,
        max_distance_neg,
        min_distance_neg,
    ) = get_min_max_distance(train_X, train_Y, best_svm)

    delta = min_distance_pos + max_distance_neg
    r = max_distance_pos + min_distance_neg
    print("delta: " + str(delta))
    print("r: ", str(r))
    print("r^2/delta^2: ", str(r**2 / delta**2))

    (
        train_distance_pos,
        train_distance_neg,
        train_distance_pos_count,
        train_distance_neg_count,
        train_mean,
    ) = split_data_get_dis_count(
        train_X, train_Y, best_svm, min_distance_pos, max_distance_neg
    )
    (
        valid_distance_pos,
        valid_distance_neg,
        valid_distance_pos_count,
        valid_distance_neg_count,
        valid_mean,
    ) = split_data_get_dis_count(
        valid_X, valid_Y, best_svm, min_distance_pos, max_distance_neg
    )
    (
        test_distance_pos,
        test_distance_neg,
        test_distance_pos_count,
        test_distance_neg_count,
        test_mean,
    ) = split_data_get_dis_count(
        test_X, test_Y, best_svm, min_distance_pos, max_distance_neg
    )

    pos_bound = min(train_distance_pos)
    neg_bound = max(train_distance_neg)
    range_list = np.arange(-10, 11, 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Train histogram")
    plt.plot(range_list, train_distance_pos_count)
    plt.plot(range_list, train_distance_neg_count)
    plt.scatter(
        train_distance_pos, [train_mean] * len(train_distance_pos), c="b", marker="."
    )
    plt.scatter(
        train_distance_neg, [train_mean] * len(train_distance_neg), c="r", marker="+"
    )
    plt.axvline(x=0, color="black")
    plt.axvline(x=-1, color="black", linestyle="--")
    plt.axvline(x=1, color="black", linestyle="--")

    plt.subplot(1, 3, 2)
    plt.title("Valid histogram")
    plt.plot(range_list, valid_distance_pos_count)
    plt.plot(range_list, valid_distance_neg_count)
    plt.scatter(
        valid_distance_pos, [valid_mean] * len(valid_distance_pos), c="b", marker="."
    )
    plt.scatter(
        valid_distance_neg, [valid_mean] * len(valid_distance_neg), c="r", marker="+"
    )
    plt.axvline(x=0, color="black")
    plt.axvline(x=pos_bound, color="black", linestyle="--")
    plt.axvline(x=neg_bound, color="black", linestyle="--")

    plt.subplot(1, 3, 3)
    plt.title("Test histogram")
    plt.plot(range_list, test_distance_pos_count)
    plt.plot(range_list, test_distance_neg_count)
    plt.scatter(
        test_distance_pos, [test_mean] * len(test_distance_pos), c="b", marker="."
    )
    plt.scatter(
        test_distance_neg, [test_mean] * len(test_distance_neg), c="r", marker="+"
    )
    plt.axvline(x=0, color="black")
    plt.axvline(x=pos_bound, color="black", linestyle="--")
    plt.axvline(x=neg_bound, color="black", linestyle="--")

    plt.savefig("./image/p1-2.png")


if __name__ == "__main__":
    n_realizations = 5
    train_size = 50
    valid_size = 50
    test_size = 1000
    evaluate_models_plot(train_size, valid_size, test_size)
