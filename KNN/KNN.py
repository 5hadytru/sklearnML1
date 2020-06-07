
import sklearn
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

dataset = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()

# Takes columns of data and assigns them integer values
buying = le.fit_transform(list(dataset['buying']))  # Returns an array of ints in this column
maint = le.fit_transform(list(dataset['maint']))
doors = le.fit_transform(list(dataset['door']))
persons = le.fit_transform(list(dataset['persons']))
lug_boot = le.fit_transform(list(dataset['lug_boot']))
safety = le.fit_transform(list(dataset['safety']))
cls = le.fit_transform(list(dataset['class']))

predict = 'class'

''' Features '''
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
''' Labels '''
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"{acc*10}%")

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(x_test)):
    print(f'Predicted: {names[predicted[x]]}', f'Data: {x_test[x]}', f'Actual: {names[y_test[x]]}')
    n = model.kneighbors([x_test[x]], 9, True)
    print(f'N: {n}')
