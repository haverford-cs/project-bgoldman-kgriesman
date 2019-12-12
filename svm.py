from data import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


#cc = ClusterCentroids(random_state=0)
#X_resampled, y_resampled = cc.fit_resample(X_resampled, y_resampled)

svm = SVC(gamma='auto', kernel="rbf")
svm.fit(X_resampled, y_resampled)
pred = svm.predict(X_test)
matrix = np.zeros((2,2))
for i in range(len(y_test)):
    true = y_test[i]
    predicted = pred[i]
    matrix[true][predicted] += 1
print(matrix)

print(svm.predict(X_test), "predict")
print(svm.score(X_test, y_test), "score")
