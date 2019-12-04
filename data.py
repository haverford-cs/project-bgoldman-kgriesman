from sklearn.impute import SimpleImputer
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
