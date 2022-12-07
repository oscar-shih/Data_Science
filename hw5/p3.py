from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

clf = SVC(kernel='linear')
X = np.array([[4, 3], [4, 8], [7, 2], [-1, -2], [-1, 3], [2, -1], [2, 1]])
Y = np.array([1, 1, 1, -1, -1, -1, -1])
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.savefig("./p3.png")