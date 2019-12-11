from data import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.svm import SVC

svm = SVC(gamma='auto', kernel="poly")
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
matrix = np.zeros((2,2))
for i in range(len(y_test)):
    true = y_test[i]
    predicted = pred[i]
    matrix[true][predicted] += 1
print(matrix)

print(svm.predict(X_test), "predict")
print(svm.score(X_test, y_test), "score")