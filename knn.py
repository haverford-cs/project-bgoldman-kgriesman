from data import data, label

from sklearn.neighbors import  KNeighborsClassifier

def main():
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(data, label)
    print(neigh.score(data, label))
    print(neigh.predict(data))

main()
