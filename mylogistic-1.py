import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import model_selection

input_file = 'new_multivar_data_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
print(X[:4, :])
print(y[:4])

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

regressor = linear_model.LogisticRegression(solver='liblinear', C=1)
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.title('Input data')
plt.show()

accuracy = 100.0 * (y_test == y_test_pred).sum() / X.shape[0]
print("Accuracy of the logistic classifier =", round(accuracy, 2), "%")
