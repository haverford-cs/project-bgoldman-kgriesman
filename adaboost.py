from data import X_train, X_test, y_train, y_test

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

adaboost = AdaBoostClassifier(n_estimators=250, random_state=0)
adaboost.fit(X_train, y_train)


y_pred = (adaboost.predict_proba(X_test)[:,1] >= 0.5).astype(int)
print(y_pred)
pred = adaboost.predict(X_test)
matrix = np.zeros((2,2))
for i in range(len(y_test)):
    true = y_test[i]
    predicted = y_pred[i]
    matrix[true][predicted] += 1
print(matrix)

#print((adaboost.predict_proba(X_test)[:,1] >= 0.4).astype(int))
print(adaboost.predict(X_test), "predict")
print(adaboost.score(X_test, y_test), "score")

#ROC curves:
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


thresh = [t for t in np.linspace(0,1,21)]
ada_tp = []
ada_fp = []
for t in thresh:
    y_pred = (adaboost.predict_proba(X_test)[:,1] >= t).astype(int)
    ada_matrix = confusion_matrix(y_test, y_pred)
    ada_fp += [false_postitive(ada_matrix)]
    ada_tp += [true_positive(ada_matrix)]

title = "ROC Curve for Ozone Days, T = 250"

plt.plot(ada_fp, ada_tp, "ro-")
plt.title(title)
plt.xlabel("false positive rates")
plt.ylabel("true positive rates")
plt.show()
