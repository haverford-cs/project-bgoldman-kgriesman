from data import X_train, X_test, y_train, y_test

from sklearn.neighbors import  KNeighborsClassifier
import numpy as np

def main():
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(X_train, y_train)
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

main()
