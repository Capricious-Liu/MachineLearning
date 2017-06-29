# from sklearn import svm
#
# X = [[2, 0], [1, 1], [2, 3]]
# y = [0, 0, 1]
#
# clf = svm.SVC(kernel='linear')
#
# clf.fit(X,y)
#
# print(clf)
#
# print(clf.support_vectors_)
#
# # index
# print(clf.support_)
#
# # numbers of support vectors in each class
# print(clf.n_support_)
#
# print(clf.predict([2,0]))




import numpy as np
import pylab as pl
from sklearn import svm

np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0]*20 + [1]*20

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a*xx -(clf.intercept_[0])/w[1]

b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1]- a*b[0])

pl.plot(xx,yy, 'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx, yy_up,'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:,1], s = 80, facecolors ='none')
pl.scatter(X[:, 0], X[:,1], c=Y, cmap = pl.cm.Paired)
pl.axis('tight')
pl.show()














