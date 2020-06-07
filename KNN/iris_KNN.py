import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import style
import math
import pickle

data = pd.read_csv('iris.data')

le = preprocessing.LabelEncoder()

# Takes columns of data and assigns them integer values
SL = le.fit_transform(list(data['SL']))  # Returns an array of ints in this column
SW = le.fit_transform(list(data['SW']))
PL = le.fit_transform(list(data['PL']))
PW = le.fit_transform(list(data['PW']))
cls = le.fit_transform(list(data['class']))

x = list(zip(SL, SW, PL, PW))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors= int(math.sqrt(len(data)) // 1))

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(f'Accuracy: {acc}\n')

with open('irismodel.pickle', 'wb') as f:
    pickle.dump(model, f)

model_in = open('irismodel.pickle', 'rb')
model = pickle.load(model_in)

predicted = model.predict(x_test)

for x in range(len(x_test)):
    print(f'\nPredicted: {predicted[x]} Data: {x_test[x]} Actual: {y_test[x]}')\

def ret_actual_values(x):
    for i in range(len(x)):
        sep_l = x[i][0] * .1 + 4.3
        sep_w = x[i][1] * .1 + 2.1
        pet_l = x[i][2] * .1 + 1
        pet_w = x[i][3] * .1 + .1
        print(sep_l, sep_w, pet_l, pet_w)