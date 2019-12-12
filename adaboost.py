from data import X_train, X_test, y_train, y_test

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids


#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

#cc = ClusterCentroids(random_state=1)
#X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
#print(len(X_resampled))


adaboost = AdaBoostClassifier(n_estimators=250, random_state=0)
adaboost.fit(X_train, y_train)



pred = adaboost.predict(X_test)
matrix = np.zeros((2,2))
for i in range(len(y_test)):
    true = y_test[i]
    predicted = pred[i]
    matrix[true][predicted] += 1
print(matrix)

print(adaboost.predict(X_test), "predict")
print(adaboost.score(X_test, y_test), "score")
