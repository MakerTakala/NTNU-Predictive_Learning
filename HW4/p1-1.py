import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def generate_data(n_samples):
    X = np.random.uniform(0, 1, (n_samples, 20))
    Y = np.sign(np.sum(X[:, :10], axis=1) - 5)
    return X, Y


def evaluate_models(n_realizations, train_size, valid_size, test_size):
    knn_errors = []
    svm_errors = []

    for _ in range(n_realizations):
        train_X, train_Y = generate_data(train_size)
        valid_X, valid_Y = generate_data(valid_size)
        test_X, test_Y = generate_data(test_size)

        best_knn_score = 0
        best_k = 1
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_X, train_Y)
            score = knn.score(valid_X, valid_Y)
            if score > best_knn_score:
                best_knn_score = score
                best_k = k

        best_knn = KNeighborsClassifier(n_neighbors=best_k)
        best_knn.fit(train_X, train_Y)
        knn_errors.append(
            {
                "train": 1 - best_knn.score(train_X, train_Y),
                "valid": 1 - best_knn_score,
                "test": 1 - best_knn.score(test_X, test_Y),
                "k": best_k,
            }
        )

        best_svm_score = 0
        best_c = 1
        for c in np.logspace(-3, 3, 10):
            svm = LinearSVC(C=c, dual=False)
            svm.fit(train_X, train_Y)
            score = svm.score(valid_X, valid_Y)
            if score > best_svm_score:
                best_svm_score = score
                best_c = c

        best_svm = LinearSVC(C=best_c, dual=False)
        best_svm.fit(train_X, train_Y)
        svm_errors.append(
            {
                "train": 1 - best_svm.score(train_X, train_Y),
                "valid": 1 - best_svm_score,
                "test": 1 - best_svm.score(test_X, test_Y),
                "c": best_c,
            }
        )

    return knn_errors, svm_errors


def plot(knn_results, svm_results):
    knn_test_errors = [result["test"] for result in knn_results]
    knn_valid_errors = [result["valid"] for result in knn_results]
    knn_train_errors = [result["train"] for result in knn_results]
    knn_best_k = [result["k"] for result in knn_results]
    svm_test_errors = [result["test"] for result in svm_results]
    svm_valid_errors = [result["valid"] for result in svm_results]
    svm_train_errors = [result["train"] for result in svm_results]
    svm_best_c = [result["c"] for result in svm_results]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    # show with boxplot for train vaild test
    plt.boxplot(
        [knn_train_errors, svm_train_errors],
        labels=["KNN", "SVM"],
    )
    plt.title("train error")
    plt.subplot(1, 3, 2)
    plt.boxplot(
        [knn_valid_errors, svm_valid_errors],
        labels=["KNN", "SVM"],
    )
    plt.title("valid error")
    plt.subplot(1, 3, 3)
    plt.boxplot(
        [knn_test_errors, svm_test_errors],
        labels=["KNN", "SVM"],
    )
    plt.title("test error")

    plt.savefig("./image/p1-1.png")


if __name__ == "__main__":
    n_realizations = 5
    train_size = 50
    valid_size = 50
    test_size = 1000
    knn_results, svm_results = evaluate_models(
        n_realizations, train_size, valid_size, test_size
    )

    plot(knn_results, svm_results)
