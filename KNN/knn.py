
from sklearn import neighbors
from sklearn import datasets
knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

print(iris)