import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Random Forest Classifier')
    parser.add_argument('--classifier', dest='classifier_type',
                        required=True, choices=['randomforest', 'extremerandomforest'],
                        help="Type of classifier to use; --classifier can be either"
                             " 'randomforest' or 'extremerandomforest'")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    print('classifier_type:' + classifier_type + '\n')

input_file = 'rforest_data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'criterion': 'entropy'}
if classifier_type == 'randomforest':
    classifier = RandomForestClassifier(**params)
else:
    classifier = ExtraTreesClassifier(**params)

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

test_datapoints = np.array([[4, 3], [2, 1], [3, 5], [8, 2], [7, 1], [5, 9]])
print("\n measure:")
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nDatapoint:', datapoint)
    print('Predicted class:', predicted_class)
    print('probabilitys"', probabilities)
