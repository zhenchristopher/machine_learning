import numpy as np
from sklearn.svm import SVC

x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])

clf = SVC(float('Inf'), 'poly', degree=2, gamma=1, coef0=1)
clf.fit(x,y)
print(clf.n_support_)