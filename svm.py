from data import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.svm import SVC

def main():
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    svm.