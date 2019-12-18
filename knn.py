from data import X_train, X_test, y_train, y_test

from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def main():
    ros = RandomOverSampler(random_state=0)
    new_x, new_y = ros.fit_resample(X_train, y_train)
    neigh = KNeighborsClassifier(n_neighbors=40)
    neigh.fit(new_x, new_y)
    print(neigh.score(X_test, y_test))
    print(neigh.predict(X_test))

    #Accuracy if predict all 0s:
    z = 0
    for i in y_test:
        if i == 0:
            z += 1
    print(z/len(y_test))

    #Confusion matrix:
    pred = neigh.predict(X_test)
    matrix = np.zeros((2,2))
    for i in range(len(y_test)):
        true = y_test[i]
        predicted = pred[i]
        matrix[true][predicted] += 1
    print(matrix)

    def confusion_matrix(y_test, y_pred):
        matrix = np.zeros((2,2))
        for i in range(len(y_test)):
            true = y_test[i]
            predicted = y_pred[i]
            matrix[true][predicted] += 1
        return matrix

    def false_postitive(conf_matrix):
        """Given a confusion matrix, returns the false positive rate. """
        fp = conf_matrix[0][1]
        total = conf_matrix[0][0]+fp
        return fp/total

    def true_positive(conf_matrix):
        """Given a confusion matrix, returns the true positive rate."""
        tp = conf_matrix[1][1]
        total = conf_matrix[1][0] + tp
        return tp/total


    """
    thresh = [t for t in np.linspace(0,1,41)]
    knn_tp = []
    knn_fp = []
    for t in thresh:
        y_pred = (neigh.predict_proba(X_test)[:,1] >= t).astype(int)
        knn_matrix = confusion_matrix(y_test, y_pred)
        knn_fp += [false_postitive(knn_matrix)]
        knn_tp += [true_positive(knn_matrix)]

    title = "ROC Curve for Ozone Days, n = 100"

    plt.plot(knn_fp, knn_tp, "ro-")
    plt.title(title)
    plt.xlabel("false positive rates")
    plt.ylabel("true positive rates")
    plt.show()
    """
main()
