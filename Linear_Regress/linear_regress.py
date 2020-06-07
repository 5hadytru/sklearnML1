
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# Split data into four arrays to train and test for G3
# x_train is part of x array, y_train is part of y array, x_test and y_test are other parts of those
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0

for i in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

# Create best fit line aka trains
    linear.fit(x_train, y_train)
# The computer predicts y_train values with x_train values;
# The acc variable returns the accuracy of these predictions
    acc = linear.score(x_test, y_test)
    print(f'Accuracy: {acc}')
# Writes/serializes linear model into a pickle file if it's the best
    if acc > best:
        acc = best
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

# Opens then loads/deserializes linear model into linear variable
pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# Printing intercepts and coefficients of best fit lines
print(f'\nCoefficients: \n{linear.coef_}')
print(f'\nIntercepts: \n{linear.intercept_}\n')

# Returns predicted y_test values from x_test values using weights gained from training
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(f'Prediction: {predictions[x]} Data: {x_test[x]} Real: {y_test[x]}')

p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()

