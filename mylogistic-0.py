import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
print(X.shape,y.shape)

classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier.fit(X, y)
y_pred = classifier.predict(X)
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the logistic classifier =", round(accuracy, 2), "%")
