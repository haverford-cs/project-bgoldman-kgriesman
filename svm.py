from data import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

svm = SVC(gamma='auto', probability=True, kernel="linear")
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

print(svm.coef_)

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



thresh = [t for t in np.linspace(0,1,41)]
svm_tp = []
svm_fp = []
for t in thresh:
    y_pred = (svm.predict_proba(X_test)[:,1] >= t).astype(int)
    svm_matrix = confusion_matrix(y_test, y_pred)
    svm_fp += [false_postitive(svm_matrix)]
    svm_tp += [true_positive(svm_matrix)]

title = "SVM ROC Curve for Ozone Days"

plt.plot(svm_fp, svm_tp, "ro-")
plt.title(title)
plt.xlabel("false positive rates")
plt.ylabel("true positive rates")
plt.show()
