from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import numpy as np

def numify(n):
    if n == '?':
        return np.nan
    else:
        return float(n)

f = open("onehr.data", "r")
label = []
data = []
for x in f:
    x = x.split(",")
    label.append(int(x[-1][0]))
    #print(x)
    x = x[1:-1]
    x = list(map(numify, x))
    data.append(x)
    #print(x)
print(data)
print(label)

data = np.asarray(data)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data)
SimpleImputer()

data = imp.transform(data)
print(imp.transform(data))

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
