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
#print(data)
#print(label)

data = np.asarray(data)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data)
SimpleImputer()

data = imp.transform(data)
#print(imp.transform(data))

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

feature_names = ['WSR0','WSR1','WSR2','WSR3','WSR4','WSR5','WSR6','WSR7','WSR8','WSR9','WSR10','WSR11','WSR12','WSR13','WSR14',
'WSR15','WSR16','WSR17','WSR18','WSR19','WSR20','WSR21','WSR22','WSR23','WSR_PK','WSR_AV','T0','T1','T2','T3','T4',
'T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','T17','T18','T19','T20','T21','T22','T23','T_PK','T_AV',
'T85','RH85','U85','V85','HT85','T70','RH70','U70','V70','HT70','T50','RH50','U50','V50','HT50','KI','TT','SLP','SLP_','Precp']

#print(len(X_train), len(y_train))
#print(len(X_test), len(y_test))
