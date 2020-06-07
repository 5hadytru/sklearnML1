import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

# x_train gets part of independent data and y_train gets part of dependent data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2)

classes = ['malignant', 'benign']

# Creates SVC object

clf = svm.SVC(kernel='linear')

# Creates SVM

clf.fit(x_train, y_train)

# Predicts y_test using x_test

y_prediction = clf.predict(x_test)

# Compares lists to return accuracy

acc = metrics.accuracy_score(y_test, y_prediction)

print(acc)
