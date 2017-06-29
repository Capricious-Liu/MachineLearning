from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# 决策树 输入格式转化 数值型
# 数据初始化
data = open("data.csv", "r")
reader = csv.reader(data)
headers = next(reader)

# print(reader)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(str(dummyX))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(str(dummyY))


# lib
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print(str(clf))


onerowX = dummyX[0, :]
print(onerowX)
testX = onerowX
testX[0] = 1
testX[2] = 0
print(testX)
predicted = clf.predict(testX)
print(predicted)